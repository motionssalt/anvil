# =====================================================================
# Anvil — Cell 2 / 2  ·  Agent interface
# ---------------------------------------------------------------------
# Launches a Gradio app that IS the entire user-facing product from
# here on. Do NOT run any more cells after this one — everything
# happens inside the Gradio UI. When you're done, interrupt this cell.
#
# The UI is a single chat stream, in the style of a standard AI agent
# interface:
#   • Tool calls stream INLINE in the chat as collapsed, expandable
#     entries (arguments + truncated result inside; click to expand).
#   • The model's reasoning appears inline, right where it happens.
#   • Delivered files appear as attachments in the chat message where
#     they are produced — each with BOTH a direct download (rendered
#     file attachment) and a tempfile.org temporary link.
#   • The download-link expiry lives in a collapsed Settings accordion,
#     not on the main screen.
#
# Under the hood: the model's NATIVE structured tool calling
# (Qwen3-VL emits <tool_call>{...}</tool_call> blocks), not a parsed
# free-text Thought/Action/Observation format. Tool results go back to
# the model as real role="tool" messages, so the agent can never
# fabricate an Observation — it only ever sees genuine tool output.
# Tools: python/shell exec, self-directed pip/apt install, file I/O,
# image viewing, web search + fetch, dual-path file delivery.
# =====================================================================

import os, sys, io, re, json, time, base64, uuid, shutil, tempfile
import subprocess, textwrap, traceback, contextlib, threading, mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator

import requests
import gradio as gr
from PIL import Image

import torch

# Pull the pre-loaded model/processor that setup.py stashes on builtins.
import builtins
if not hasattr(builtins, "ANVIL"):
    raise RuntimeError("Model not loaded. Run `anvil.setup` before launching the agent.")
ANVIL = builtins.ANVIL

model      = ANVIL["model"]
processor  = ANVIL["processor"]
MODEL_ID   = ANVIL["model_id"]
GPU_NAME   = ANVIL["gpu_name"]
VRAM_GB    = ANVIL["vram_gb"]
MAX_NEW    = ANVIL["max_new_tokens"]

# ---------------------------------------------------------------------
# Workspace inside the Colab VM
# ---------------------------------------------------------------------
WORKDIR = Path(os.environ.get("ANVIL_WORKDIR", "/content/anvil_workspace"))
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

# ---------------------------------------------------------------------
# Gradio compatibility
# ---------------------------------------------------------------------
# The inline tool-call UI needs the messages-format Chatbot (metadata
# titles render as collapsible entries). gradio>=4.44 supports it; if a
# runtime ships something older we degrade to plain text in tuples.
import inspect

_CHATBOT_PARAMETERS = inspect.signature(gr.Chatbot.__init__).parameters
_CHATBOT_SUPPORTS_TYPE = "type" in _CHATBOT_PARAMETERS
try:
    _GRADIO_MAJOR = int(str(getattr(gr, "__version__", "0")).split(".")[0])
except (TypeError, ValueError):
    _GRADIO_MAJOR = 0
_CHATBOT_USES_MESSAGES = _CHATBOT_SUPPORTS_TYPE or _GRADIO_MAJOR >= 5
_HAS_MULTIMODAL_TEXTBOX = hasattr(gr, "MultimodalTextbox")


def _chatbot_history_initial():
    if _CHATBOT_USES_MESSAGES:
        return []
    return []

# ---------------------------------------------------------------------
# TempFile.org uploader — verified against https://tempfile.org/api
# ---------------------------------------------------------------------
TEMPFILE_API   = "https://tempfile.org/api/upload/local"
TEMPFILE_LIMIT = 100 * 1024 * 1024   # 100 MB per file, per the docs
VALID_EXPIRY   = (1, 6, 24, 48)
DEFAULT_EXPIRY = 6

def tempfile_upload(path: str, expiry_hours: int = DEFAULT_EXPIRY) -> Dict[str, Any]:
    """
    Upload a single file to tempfile.org and return
        {"ok": True, "url": "...", "expiry_hours": N}
    or
        {"ok": False, "error": "..."}.
    """
    p = Path(path)
    if not p.exists():
        return {"ok": False, "error": f"no such file: {path}"}
    size = p.stat().st_size
    if size > TEMPFILE_LIMIT:
        return {"ok": False,
                "error": f"file is {size/1e6:.1f} MB, exceeds tempfile.org's 100 MB limit"}
    if expiry_hours not in VALID_EXPIRY:
        expiry_hours = DEFAULT_EXPIRY
    try:
        with open(p, "rb") as fh:
            r = requests.post(
                TEMPFILE_API,
                files={"files": (p.name, fh)},
                data={"expiryHours": str(expiry_hours)},
                timeout=120,
            )
        r.raise_for_status()
        data = r.json()
        files = data.get("files") or []
        if not files or "url" not in files[0]:
            return {"ok": False, "error": f"unexpected tempfile.org response: {data}"}
        return {"ok": True, "url": files[0]["url"], "expiry_hours": expiry_hours}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# ---------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------
# Persistent Python namespace so successive `run_python` calls share state.
_PY_NS: Dict[str, Any] = {"__name__": "__anvil__"}

def tool_run_python(code: str) -> str:
    """Execute python in a persistent namespace. Return stdout + a short repr of the last expression."""
    buf = io.StringIO()
    err = None
    last_val = None
    try:
        tree_src = code
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                import ast
                tree = ast.parse(tree_src, mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body[-1].value)
                    exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]),
                                 "<anvil>", "exec"), _PY_NS)
                    last_val = eval(compile(last, "<anvil>", "eval"), _PY_NS)
                else:
                    exec(compile(tree, "<anvil>", "exec"), _PY_NS)
            except SystemExit:
                pass
    except Exception:
        err = traceback.format_exc(limit=4)
    out = buf.getvalue()
    if last_val is not None:
        out += ("\n" if out and not out.endswith("\n") else "") + repr(last_val)
    if err:
        out += ("\n" if out and not out.endswith("\n") else "") + err
    return (out or "(no output)").strip()[:8000]

def tool_run_shell(cmd: str, timeout: int = 300) -> str:
    """Run a shell command inside the Colab VM. Combined stdout+stderr."""
    try:
        r = subprocess.run(
            cmd, shell=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        out = r.stdout or ""
        tag = f"[exit {r.returncode}] "
        return (tag + out).strip()[:8000] or f"[exit {r.returncode}] (no output)"
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

def tool_read_file(path: str, max_bytes: int = 20000) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"[error] no such file: {path}"
    if p.is_dir():
        return "\n".join(str(x) for x in p.iterdir())
    try:
        data = p.read_bytes()
    except Exception as e:
        return f"[error] {e}"
    truncated = ""
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = f"\n\n[…truncated, file is {p.stat().st_size} bytes total…]"
    try:
        return data.decode("utf-8") + truncated
    except UnicodeDecodeError:
        return f"[binary file, {p.stat().st_size} bytes]"

def tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p} ({p.stat().st_size} bytes)"

# Web search + fetch --------------------------------------------------
def tool_web_search(query: str, k: int = 6) -> str:
    try:
        from duckduckgo_search import DDGS
    except Exception as e:
        return f"[error importing duckduckgo_search: {e}]"
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=k))
    except Exception as e:
        return f"[search error] {type(e).__name__}: {e}"
    if not hits:
        return "(no results)"
    lines = []
    for i, h in enumerate(hits, 1):
        title = h.get("title") or ""
        url   = h.get("href")  or h.get("url") or ""
        body  = (h.get("body") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {title}\n   {url}\n   {body[:240]}")
    return "\n".join(lines)

def tool_fetch_url(url: str, max_chars: int = 6000) -> str:
    try:
        r = requests.get(
            url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Anvil; +https://tempfile.org)"},
        )
        r.raise_for_status()
    except Exception as e:
        return f"[fetch error] {type(e).__name__}: {e}"
    ctype = r.headers.get("Content-Type", "")
    if "html" in ctype or url.endswith((".html", ".htm", "/")):
        try:
            from readability import Document
            from markdownify import markdownify as md
            doc = Document(r.text)
            title = doc.short_title()
            body_md = md(doc.summary(html_partial=True))
            text = f"# {title}\n\n{body_md}"
        except Exception:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "lxml")
            for t in soup(["script", "style", "nav", "footer"]):
                t.decompose()
            text = soup.get_text("\n", strip=True)
    else:
        text = r.text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[…truncated, page was {len(r.text)} chars…]"
    return text

# Image viewing ------------------------------------------------------
# view_image registers the image; the loop then appends it to the
# message stream as real vision input, so the model literally sees it
# on the next forward pass.
_VIEWED_IMAGES: Dict[str, str] = {}  # tag -> path

def tool_view_image(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"[error] no such image: {path}"
    try:
        img = Image.open(p)
        w, h = img.size
    except Exception as e:
        return f"[error opening image: {e}]"
    tag = f"img_{len(_VIEWED_IMAGES)+1}"
    _VIEWED_IMAGES[tag] = str(p)
    return (f"loaded image '{p.name}' as {tag}  ({w}×{h}). "
            f"The image is now in your vision context — describe or reason "
            f"about its actual contents in your next step.")

# File delivery ------------------------------------------------------
# deliver_file hands a file back through BOTH a direct download in the
# chat AND a tempfile.org temporary link, inline in the message stream.
_DELIVERED: List[Dict[str, Any]] = []
_TEMPFILE_EXPIRY: int = DEFAULT_EXPIRY  # set from the Settings accordion

def tool_deliver_file(path: str, description: str = "") -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return f"[error] no such file to deliver: {path}"
    up = tempfile_upload(str(p), expiry_hours=_TEMPFILE_EXPIRY)
    rec = {
        "path": str(p),
        "name": p.name,
        "size": p.stat().st_size,
        "description": description or "",
        "tempfile": up,
    }
    _DELIVERED.append(rec)
    if up.get("ok"):
        return (f"delivered '{p.name}' ({p.stat().st_size} bytes). "
                f"Temp link ({up['expiry_hours']}h): {up['url']}  "
                f"(also attached inline in the chat as a direct download)")
    else:
        return (f"delivered '{p.name}' ({p.stat().st_size} bytes) as a direct "
                f"download in the chat. tempfile.org upload failed: {up.get('error')}")

# ---------------------------------------------------------------------
# Tool registry — names + implementations
# ---------------------------------------------------------------------
TOOLS: Dict[str, Dict[str, Any]] = {
    "run_python": {"fn": tool_run_python, "args": ["code"]},
    "run_shell": {"fn": tool_run_shell, "args": ["cmd"]},
    "read_file": {"fn": tool_read_file, "args": ["path"]},
    "write_file": {"fn": tool_write_file, "args": ["path", "content"]},
    "view_image": {"fn": tool_view_image, "args": ["path"]},
    "web_search": {"fn": tool_web_search, "args": ["query"]},
    "fetch_url": {"fn": tool_fetch_url, "args": ["url"]},
    "deliver_file": {"fn": tool_deliver_file, "args": ["path", "description"]},
}

# OpenAI-style tool schemas for the model's NATIVE tool calling. The
# chat template renders these into the prompt and parses the model's
# <tool_call> output back into structured calls — no free-text ReAct.
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": ("Execute Python code in a persistent notebook-like namespace. "
                        "State (variables, imports) persists between calls. Use this for "
                        "computation, data processing, file manipulation, plotting, etc."),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string", "description": "Python source to execute."}},
                       "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": ("Run a shell command in the Colab VM. Full sudo is available. "
                        "Use `pip install …` or `apt-get install -y …` freely to install "
                        "any dependency you need mid-task."),
        "parameters": {"type": "object",
                       "properties": {
                           "cmd": {"type": "string", "description": "Shell command to run."},
                           "timeout": {"type": "integer", "description": "Timeout in seconds (default 300)."}},
                       "required": ["cmd"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file (or list a directory) from the Colab VM.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write text content to a file in the Colab VM.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "view_image",
        "description": ("Load an image file so you can look at it directly with your vision "
                        "capabilities. After calling this, the image enters your vision "
                        "context and you can describe or reason about its actual contents."),
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": ("Search the web (DuckDuckGo). Returns titles + URLs + snippets. "
                        "Use this whenever you need current info or aren't sure of a fact."),
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch a URL and return its readable text content (HTML is converted to markdown).",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "deliver_file",
        "description": ("Hand a produced file back to the user. This gives them BOTH a "
                        "direct download attached inline in the chat AND a temporary "
                        "tempfile.org link. Call this for every output file the user "
                        "should receive."),
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "description": {"type": "string",
                                                      "description": "One-line human summary of the file."}},
                       "required": ["path"]}}},
]

# ---------------------------------------------------------------------
# System prompt — hard behavioral rules
# ---------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are **Anvil**, an autonomous tool-using AI agent running
inside a Google Colab notebook on a {GPU_NAME} GPU with about {VRAM_GB:.0f} GB
of VRAM. You have full control of the Colab VM: you can install packages
(pip / apt), run arbitrary Python and shell, read/write files, search the
web, and look at images directly with your own vision.

## Environment
- OS: Linux (Colab). Working directory: {WORKDIR}.
- You may install anything you need with `pip install …` or
  `apt-get install -y …` via the `run_shell` tool — do this proactively
  when a task needs it, without asking the user first.
- You are on a shared GPU. Be mindful of VRAM in code you run.
- Colab sessions are NOT permanent. Anything the user needs to keep must
  be delivered back via the `deliver_file` tool while the session is live.

## Hard rules — these are mandatory, not suggestions
1. NEVER conclude that a resource, repo, file, or URL does not exist based
   on a web search alone. Search engines miss things. If the task would be
   settled by a direct check, DO the direct check: clone it
   (`run_shell` with `git clone …`), probe it (`git ls-remote`, `curl -I`),
   or fetch it (`fetch_url`) — then report what the direct check found.
2. NEVER fabricate, invent, or guess the result of a tool call. You only
   ever see real tool output, delivered to you as tool-result messages.
   If you have not called the tool yet, you do not know its result — call
   it and wait.
3. If a task needs a CLI tool or library that is not installed (ffmpeg,
   mpv, imagemagick, yt-dlp, a pip package, …), INSTALL it yourself with
   `run_shell` and then proceed. "Tool X is not installed" is NEVER an
   acceptable final answer.
4. If one tool fails or returns nothing useful, TRY ANOTHER approach
   before giving up: a different search query, `fetch_url` on a direct
   URL, `run_shell` with a CLI equivalent, or `run_python`. Give up on a
   sub-goal only after at least two genuinely different attempts have
   failed — and say exactly what you tried.
5. When you produce a file for the user, ALWAYS call `deliver_file(path)`
   — do not just leave it in the filesystem.
6. Uploaded files (if any) are listed at the start of the task. An image
   the user attached is already in your vision context — you may describe
   it directly. To look at an image you produced or downloaded yourself,
   call `view_image(path)` first.
7. When the task is complete, reply to the user with a concise final
   message and make no further tool calls.
"""

# ---------------------------------------------------------------------
# LLM call — native tool calling
# ---------------------------------------------------------------------
def _llm_call(messages: List[Dict[str, Any]]) -> str:
    """One forward pass through the model. Returns the newly generated text."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        tools=TOOL_SCHEMAS,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()

# ---------------------------------------------------------------------
# Native tool-call parsing
# ---------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _extract_json_block(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the first balanced {...} block out of a string."""
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except Exception:
                    return None
    return None


def _parse_model_output(raw: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Split raw model output into (reasoning_text, [tool_calls]).
    Each tool call is {"id": ..., "name": ..., "arguments": {...}}.
    Tolerates bare JSON dumps (no <tool_call> wrapper) as a fallback.
    """
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(raw):
        block = _extract_json_block(m.group(1))
        if block and block.get("name"):
            args = block.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}",
                          "name": str(block["name"]),
                          "arguments": args})
    reasoning = _TOOL_CALL_RE.sub("", raw).strip()

    if not calls and '"name"' in raw:
        # Fallback: the model emitted bare tool-call JSON without wrappers.
        block = _extract_json_block(raw)
        if block and block.get("name"):
            args = block.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": str(args)}
            calls.append({"id": f"call_{uuid.uuid4().hex[:8]}",
                          "name": str(block["name"]),
                          "arguments": args})
            reasoning = raw[:raw.find("{")].strip()
    return reasoning, calls


def _dispatch_tool(name: str, args: Dict[str, Any]) -> str:
    """Execute one tool call and return its real output as a string."""
    if name not in TOOLS:
        return f"[error] unknown tool: {name}. Available: {', '.join(TOOLS)}"
    spec = TOOLS[name]
    try:
        call_args = {}
        for arg_name in spec["args"]:
            if arg_name in args:
                call_args[arg_name] = args[arg_name]
        # Fallback: if the model dumped a raw string, use it as the
        # first positional arg.
        if not call_args and "_raw" in args and spec["args"]:
            call_args[spec["args"][0]] = args["_raw"]
        return spec["fn"](**call_args)
    except Exception as e:
        return f"[tool error] {type(e).__name__}: {e}"

# ---------------------------------------------------------------------
# UI rendering helpers — everything lives in the chat stream
# ---------------------------------------------------------------------
def _preview(text: str, n: int = 160) -> str:
    text = " ".join((text or "").split())
    return text[:n] + ("…" if len(text) > n else "")


def _tool_title(name: str, args: Dict[str, Any], result: Optional[str]) -> str:
    arg_bits = ", ".join(f"{k}={_preview(str(v), 40)!r}" for k, v in list(args.items())[:2])
    title = f"🛠 {name}({arg_bits})"
    if result is not None:
        title += f" → {_preview(result, 80)}"
    return title


def _tool_body(name: str, args: Dict[str, Any], result: Optional[str]) -> str:
    parts = [f"**arguments**\n```json\n{json.dumps(args, ensure_ascii=False, indent=2)[:2000]}\n```"]
    if result is not None:
        parts.append(f"**result**\n```\n{result[:3000]}\n```")
    else:
        parts.append("*running…*")
    return "\n\n".join(parts)


def _render_tool_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """A tool call as one collapsible chat entry (metadata title = header)."""
    return {
        "role": "assistant",
        "content": _tool_body(evt["name"], evt["args"], evt.get("result")),
        "metadata": {"title": _tool_title(evt["name"], evt["args"], evt.get("result"))},
    }


def _render_reasoning(text: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": text,
        "metadata": {"title": f"💭 {_preview(text, 90)}"},
    }


def _render_file_delivery(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A delivered file as an inline chat attachment with the temp link.

    Returns TWO messages (both contract violations lived here, and the
    split fixes each):

    1. caption message — plain text content carrying the filename, size,
       description and tempfile.org link. Text content is valid on every
       messages-format Gradio version. (Previously this caption lived in
       the ``alt_text`` of a raw ``{"path": ..., "alt_text": ...}`` dict
       used as message ``content`` — a shape the messages-format Chatbot
       does not guarantee: ``_postprocess_content`` only explicitly
       handles ``str`` / ``FileData`` / ``GradioComponent`` /
       ``(path, alt_text)`` tuples, and in several 4.x versions a raw
       dict falls through and postprocessing yields ``None`` for the
       message, which surfaces downstream as
       ``Value after * must be an iterable, not NoneType``.)

    2. attachment message — ``content`` is a ``gr.FileData`` object (the
       documented, version-stable file-content shape; handled via an
       explicit ``isinstance(chat_message, FileData)`` branch in
       ``Chatbot._postprocess_content`` across the 4.x/5.x/6.x line) and
       carries NO ``metadata``. Previously the same message combined
       file content with ``metadata={"title": ...}``; in Gradio's
       messages format ``metadata`` is the collapsible-header mechanism
       intended for text messages, and combining it with file content is
       version-dependent and can silently fail postprocessing (again a
       ``None`` message downstream). No functionality is lost: the
       caption message above shows everything the metadata title showed.
    """
    caption_lines = [f"📎 **Delivered: {rec['name']}** ({rec['size']:,} bytes)"]
    if rec.get("description"):
        caption_lines.append(rec["description"])
    up = rec.get("tempfile") or {}
    if up.get("ok"):
        caption_lines.append(f"🔗 [tempfile.org link]({up['url']}) — expires in {up['expiry_hours']}h")
    else:
        caption_lines.append(f"*(tempfile.org link unavailable: {up.get('error', 'n/a')})*")
    caption_msg = {
        "role": "assistant",
        "content": "\n".join(caption_lines),
    }
    attachment_msg = {
        "role": "assistant",
        "content": gr.FileData(
            path=rec["path"],
            orig_name=rec["name"],
            size=rec.get("size"),
            mime_type=mimetypes.guess_type(rec["path"])[0],
        ),
    }
    return [caption_msg, attachment_msg]


def _ui_append(ui_events: List[Dict[str, Any]], msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return ui_events + [msg]


# ---------------------------------------------------------------------
# The core streaming agent loop (native tool calling)
# ---------------------------------------------------------------------
MAX_STEPS = 24


def agent_stream(user_text: str,
                 uploaded_paths: List[str],
                 history_msgs: List[Dict[str, Any]],
                 expiry_hours: int
                 ) -> Generator[Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]], None, None]:
    """
    Yields (ui_events, final_answer_or_None, updated_agent_history).
    ui_events are chat messages to append after the user's message:
    reasoning blocks, collapsible tool calls, inline file attachments,
    and finally the assistant's reply.
    """
    global _DELIVERED, _VIEWED_IMAGES, _TEMPFILE_EXPIRY
    _DELIVERED = []
    _VIEWED_IMAGES = {}
    _TEMPFILE_EXPIRY = expiry_hours if expiry_hours in VALID_EXPIRY else DEFAULT_EXPIRY

    # Sort uploads into images vs everything else — images go into vision.
    image_paths = []
    for p in uploaded_paths:
        mime, _ = mimetypes.guess_type(p)
        if (mime or "").startswith("image/"):
            image_paths.append(p)

    content: List[Dict[str, Any]] = []
    for img_path in image_paths:
        content.append({"type": "image", "image": img_path})
    prose = user_text
    if uploaded_paths:
        listing = "\n".join(f"  • {f}" for f in uploaded_paths)
        prose += f"\n\n[The user attached the following files, saved to disk:\n{listing}\n]"
    content.append({"type": "text", "text": prose})

    # The live conversation for the model: real roles, real tool results.
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        *history_msgs,
        {"role": "user", "content": content},
    ]

    ui_events: List[Dict[str, Any]] = []
    delivered_this_turn: List[Dict[str, Any]] = []

    for step_i in range(1, MAX_STEPS + 1):
        try:
            raw = _llm_call(messages)
        except Exception as e:
            err = f"❌ LLM error: {type(e).__name__}: {e}"
            ui_events = _ui_append(ui_events, {"role": "assistant", "content": err})
            yield ui_events, None, history_msgs
            return

        reasoning, calls = _parse_model_output(raw)

        # ---------------------- no tool calls → final answer
        if not calls:
            final = (reasoning or raw or "(the model produced no answer)").strip()
            ui_events = _ui_append(ui_events, {"role": "assistant", "content": final})
            new_history = history_msgs + [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": final}]},
            ]
            yield ui_events, final, new_history
            return

        # ---------------------- record the reasoning inline, where it happened
        if reasoning:
            ui_events = _ui_append(ui_events, _render_reasoning(reasoning))

        # Append the assistant turn (with its tool calls) to the model convo.
        messages.append({
            "role": "assistant",
            "content": ([{"type": "text", "text": reasoning}] if reasoning else []),
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in calls
            ],
        })

        # ---------------------- execute each tool call, stream results inline
        tool_result_msgs: List[Dict[str, Any]] = []
        for call in calls:
            evt = {"name": call["name"], "args": call["arguments"], "result": None}
            rendered = _render_tool_event(evt)
            ui_events = _ui_append(ui_events, rendered)
            yield ui_events, None, history_msgs  # show the call as "running…"

            result = _dispatch_tool(call["name"], call["arguments"])

            evt["result"] = result
            ui_events[-1] = _render_tool_event(evt)  # update in place with the result
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": [{"type": "text", "text": result}],
            })

            # If a file was delivered, attach it inline right here.
            if call["name"] == "deliver_file" and _DELIVERED:
                rec = _DELIVERED[-1]
                if rec not in delivered_this_turn:
                    delivered_this_turn.append(rec)
                    # _render_file_delivery now returns a list of messages
                    # (text caption + clean file attachment).
                    ui_events = ui_events + _render_file_delivery(rec)

            yield ui_events, None, history_msgs

        # Feed the REAL tool results back to the model. The agent can
        # never fabricate an Observation — these are genuine outputs.
        messages.extend(tool_result_msgs)

        # If view_image ran, inject the image(s) as real vision input.
        new_imgs = [p for p in _VIEWED_IMAGES.values()
                    if not any(p in json.dumps(m, default=str) for m in messages[:-len(tool_result_msgs)])]
        if new_imgs:
            img_content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in new_imgs]
            img_content.append({"type": "text",
                                "text": "[The image(s) you loaded with view_image are above. "
                                        "Describe what you actually see.]"})
            messages.append({"role": "user", "content": img_content})

    # Loop exhausted.
    final = ("I hit the step limit before finishing. Here's where I got to — "
             "ask a narrower follow-up and I'll continue from here.")
    ui_events = _ui_append(ui_events, {"role": "assistant", "content": final})
    new_history = history_msgs + [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": final}]},
    ]
    yield ui_events, final, new_history


# ---------------------------------------------------------------------
# Gradio glue
# ---------------------------------------------------------------------
def _copy_uploads(files) -> List[str]:
    """Copy Gradio-uploaded files into the workspace and return their paths."""
    saved = []
    if not files:
        return saved
    up_dir = WORKDIR / "uploads"
    up_dir.mkdir(exist_ok=True)
    for f in files:
        src = getattr(f, "name", None) or f
        src = str(src)
        dest = up_dir / f"{uuid.uuid4().hex[:8]}_{Path(src).name}"
        try:
            shutil.copy(src, dest)
            saved.append(str(dest))
        except Exception:
            pass
    return saved


def _expiry_from_label(label: str) -> int:
    try:
        val = int(str(label).split()[0])
        return val if val in VALID_EXPIRY else DEFAULT_EXPIRY
    except Exception:
        return DEFAULT_EXPIRY


# ---------- Gradio callback (a generator, for live streaming) ----------
def on_submit(user_msg: str,
              files,
              expiry_choice: str,
              chat_history,
              agent_history: List[Dict[str, Any]]):
    """Streaming handler. Yields updates to
    (chatbot, agent_history_state, file_input, input_box)."""
    user_msg = (user_msg or "").strip()
    uploads = _copy_uploads(files)
    if not uploads:
        uploads = [str(Path(p)) for p in getattr(builtins, "ANVIL_FILES", [])
                   if Path(p).exists()]
    if not user_msg and not uploads:
        yield chat_history, agent_history, None, ""
        return
    if not user_msg and uploads:
        user_msg = "(no text — please look at the uploaded file(s) and decide what to do.)"

    # Append the user turn to the visible chat immediately.
    display_user = user_msg
    if uploads:
        display_user += "\n\n" + "\n".join(f"📎 `{Path(p).name}`" for p in uploads)

    if _CHATBOT_USES_MESSAGES:
        chat_history = chat_history + [{"role": "user", "content": display_user}]
    else:
        chat_history = chat_history + [[display_user, None]]
    yield chat_history, agent_history, None, ""

    expiry = _expiry_from_label(expiry_choice)
    new_agent_history = agent_history

    for ui_events, final, hist in agent_stream(user_msg, uploads, agent_history, expiry):
        if _CHATBOT_USES_MESSAGES:
            visible = chat_history + ui_events
        else:
            # Legacy tuple fallback: flatten events into text.
            lines = []
            for ev in ui_events:
                title = (ev.get("metadata") or {}).get("title", "")
                body = ev.get("content")
                if isinstance(body, dict):
                    body = body.get("alt_text", "")
                elif isinstance(body, gr.FileData):
                    body = f"📎 {body.orig_name or Path(body.path).name}"
                lines.append(f"{title}\n{body}" if title else str(body))
            visible = chat_history[:-1] + [[chat_history[-1][0], "\n\n".join(lines)]]
        yield visible, hist, None, ""
        if final is not None:
            new_agent_history = hist

    yield chat_history, new_agent_history, None, ""


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
_CSS = """
.gradio-container { max-width: 900px !important; margin: auto !important; }
#anvil-chatbot { border: 1px solid #e4e4e7; border-radius: 12px; }
footer { display: none !important; }
"""


def launch():
    with gr.Blocks(title="Anvil") as demo:
        gr.Markdown("## ⚒️ Anvil")

        agent_hist_state = gr.State([])

        chatbot_params = {
            "type": "messages",
            "height": 560,
            "show_label": False,
            "show_copy_button": True,
            "render_markdown": True,
            "value": _chatbot_history_initial(),
            "elem_id": "anvil-chatbot",
        }
        chatbot = gr.Chatbot(**{
            k: v for k, v in chatbot_params.items() if k in _CHATBOT_PARAMETERS
        })

        if _HAS_MULTIMODAL_TEXTBOX:
            input_box = gr.MultimodalTextbox(
                placeholder="What would you like Anvil to do? Attach files with 📎",
                file_count="multiple",
                show_label=False,
                submit_btn=True,
            )
            file_input = None
        else:
            input_box = gr.Textbox(
                placeholder="What would you like Anvil to do?",
                lines=2, show_label=False,
            )
            file_input = gr.File(label="Attachments", file_count="multiple")

        with gr.Accordion("Settings", open=False):
            expiry_dd = gr.Dropdown(
                choices=[f"{h} hours" for h in VALID_EXPIRY],
                value=f"{DEFAULT_EXPIRY} hours",
                label="Temp-file link expiry",
            )

        def _unpack(msg_value):
            """MultimodalTextbox returns {'text': ..., 'files': [...]}."""
            if isinstance(msg_value, dict):
                return msg_value.get("text", ""), msg_value.get("files", [])
            return msg_value or "", None

        def _submit_norm(msg_value, expiry, chat, agent_hist):
            """MultimodalTextbox path: text AND files both arrive inside msg_value."""
            text, mm_files = _unpack(msg_value)
            all_files = list(mm_files or [])
            for chat_out, hist_out, _f, _t in on_submit(text, all_files, expiry, chat, agent_hist):
                yield chat_out, hist_out, {"text": "", "files": []}

        def _submit_with_files(msg_value, files, expiry, chat, agent_hist):
            """Plain Textbox path: files come from the separate file_input component."""
            text, _ = _unpack(msg_value)
            all_files = list(files or [])
            for chat_out, hist_out, _f, _t in on_submit(text, all_files, expiry, chat, agent_hist):
                yield chat_out, hist_out, None, ""

        if file_input is None:
            input_box.submit(_submit_norm,
                             inputs=[input_box, expiry_dd, chatbot, agent_hist_state],
                             outputs=[chatbot, agent_hist_state, input_box])
        else:
            input_box.submit(_submit_with_files,
                             inputs=[input_box, file_input, expiry_dd, chatbot, agent_hist_state],
                             outputs=[chatbot, agent_hist_state, file_input, input_box])

        def _clear():
            return _chatbot_history_initial(), []

        chatbot.clear(_clear, outputs=[chatbot, agent_hist_state])

    print(f"Launching Anvil. Model: {MODEL_ID}. GPU: {GPU_NAME}.")
    try:
        queued_demo = demo.queue(default_concurrency_limit=1)
    except TypeError:
        queued_demo = demo.queue()
    queued_demo.launch(share=True, debug=False, inline=True, show_error=True, css=_CSS)


if __name__ == "__main__":
    launch()
