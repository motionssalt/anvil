# =====================================================================
# Anvil — Cell 2 / 2  ·  Agent interface
# ---------------------------------------------------------------------
# Launches a Gradio app that IS the entire user-facing product from
# here on. Do NOT run any more cells after this one — everything
# happens inside the Gradio UI. When you're done, interrupt this cell.
#
# The Gradio UI has:
#   • A chat pane (with file-upload paperclip)
#   • A "🧠 Thought process" pane that updates live as the agent works
#   • A file-output area for anything the agent produces
#   • A "TempFile.org expiry" dropdown so you can override per-run
#
# Under the hood: a hand-rolled ReAct loop feeding a vision-capable
# tool-calling LLM. Tools include python/shell exec, self-directed
# `pip`/`apt` install, file I/O, image viewing, web search + fetch,
# and a `deliver_file` tool that hands files back through BOTH the
# Gradio download AND a tempfile.org temporary link.
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
WORKDIR = Path("/content/anvil_workspace")
WORKDIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORKDIR)

# ---------------------------------------------------------------------
# Gradio compatibility
# ---------------------------------------------------------------------
# Gradio 4/5 commonly supports Chatbot(type="messages"), while some
# Colab runtimes ship a newer or older Chatbot API. Keep the UI and
# callback history format aligned with the installed constructor.
import inspect

_CHATBOT_SUPPORTS_MESSAGES = "type" in inspect.signature(gr.Chatbot.__init__).parameters

def _chatbot_history_initial():
    if _CHATBOT_SUPPORTS_MESSAGES:
        return [{"role": "assistant", "content": "Anvil is ready."}]
    return [[None, "Anvil is ready."]]

def _chatbot_history_add_turn(history, user_text):
    if _CHATBOT_SUPPORTS_MESSAGES:
        return history + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": "*thinking…*"},
        ]
    return history + [[user_text, "*thinking…*"]]

def _chatbot_history_set_assistant(history, text):
    if _CHATBOT_SUPPORTS_MESSAGES:
        history[-1] = {"role": "assistant", "content": text}
    else:
        history[-1][1] = text
    return history

# ---------------------------------------------------------------------
# TempFile.org uploader — verified against https://tempfile.org/api
# ---------------------------------------------------------------------
TEMPFILE_API   = "https://tempfile.org/api/upload/local"
TEMPFILE_LIMIT = 100 * 1024 * 1024   # 100 MB per file, per the docs
VALID_EXPIRY   = (1, 6, 24, 48)

def tempfile_upload(path: str, expiry_hours: int = 1) -> Dict[str, Any]:
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
        expiry_hours = 1
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
        # Try to split trailing expression for repr
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
# view_image just loads the image; the *actual* multimodal understanding
# happens because we re-inject the referenced image into the next LLM
# call as vision input. See _build_prompt() below.
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
            f"You can now describe or reason about its contents directly — "
            f"the image is included in your vision context.")

# File delivery ------------------------------------------------------
# The agent calls `deliver_file(path)` when it produces a file. The
# main event loop watches for these tool calls and, after the loop
# finishes, presents ALL delivered files via BOTH the Gradio file-
# output component AND their tempfile.org URLs.
_DELIVERED: List[Dict[str, Any]] = []
_TEMPFILE_EXPIRY: int = 1  # updated from the UI dropdown per-run

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
                f"(also available as direct download in the UI)")
    else:
        return (f"delivered '{p.name}' ({p.stat().st_size} bytes) as direct "
                f"download in the UI. tempfile.org upload failed: {up.get('error')}")

# ---------------------------------------------------------------------
# Tool registry — the LLM sees these descriptions verbatim.
# ---------------------------------------------------------------------
TOOLS: Dict[str, Dict[str, Any]] = {
    "run_python": {
        "fn": tool_run_python,
        "args": ["code"],
        "desc": "Execute Python code in a persistent notebook-like namespace. "
                "State (variables, imports) persists between calls. Use this for "
                "computation, data processing, file manipulation, plotting, etc.",
    },
    "run_shell": {
        "fn": tool_run_shell,
        "args": ["cmd"],
        "desc": "Run a shell command in the Colab VM. Full sudo is available. "
                "Use `pip install …` or `apt-get install -y …` freely to install "
                "any dependency you need mid-task.",
    },
    "read_file": {
        "fn": tool_read_file,
        "args": ["path"],
        "desc": "Read a text file (or list a directory) from the Colab VM.",
    },
    "write_file": {
        "fn": tool_write_file,
        "args": ["path", "content"],
        "desc": "Write text content to a file in the Colab VM.",
    },
    "view_image": {
        "fn": tool_view_image,
        "args": ["path"],
        "desc": "Load an image so you can look at it directly with your vision "
                "capabilities. After calling this, you can describe or reason "
                "about the image in your next thought.",
    },
    "web_search": {
        "fn": tool_web_search,
        "args": ["query"],
        "desc": "Search the web (DuckDuckGo). Returns titles + URLs + snippets. "
                "Use this whenever you need current info or aren't sure of a fact.",
    },
    "fetch_url": {
        "fn": tool_fetch_url,
        "args": ["url"],
        "desc": "Fetch a URL and return its readable text content (HTML is "
                "converted to markdown).",
    },
    "deliver_file": {
        "fn": tool_deliver_file,
        "args": ["path", "description"],
        "desc": "Hand a produced file back to the user. This gives them BOTH a "
                "direct download in the UI AND a temporary tempfile.org link. "
                "Call this for every output file the user should receive. "
                "`description` is a one-line human summary of what the file is.",
    },
    "final_answer": {
        "fn": None,   # handled specially
        "args": ["answer"],
        "desc": "Emit the final clean answer to the user and stop. Do this once "
                "the task is complete.",
    },
}

def _tool_descriptions_for_prompt() -> str:
    out = []
    for name, spec in TOOLS.items():
        args = ", ".join(spec["args"])
        out.append(f"- {name}({args}): {spec['desc']}")
    return "\n".join(out)

# ---------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are **Anvil**, an autonomous tool-using AI agent running
inside a Google Colab notebook on a {GPU_NAME} GPU with about {VRAM_GB:.0f} GB
of VRAM. You have full control of the Colab VM: you can install packages
(pip / apt), run arbitrary Python and shell, read/write files, search the
web, and look at images directly with your own vision.

## Environment
- OS: Linux (Colab). Working directory: /content/anvil_workspace.
- You may install anything you need with `pip install …` or `apt-get install -y …`
  via the `run_shell` tool — do this proactively when a task needs it, don't
  ask the user first.
- You are on a shared GPU. Be mindful of VRAM in code you run (don't load
  giant models unless the task genuinely requires it).
- Colab sessions are NOT permanent. Anything the user needs to keep must be
  delivered back to them via the `deliver_file` tool while the session is live.
- When you produce a file for the user, ALWAYS call `deliver_file(path)` —
  do not just leave it in the filesystem. `deliver_file` gives them both a
  direct download and a temporary tempfile.org link.

## How you work — ReAct loop
Every step you produce EXACTLY ONE of the following, and nothing else:

    Thought: <one short line describing what you're about to do or figure out>
    Action: <tool_name>
    Action Input: <JSON object with the tool's arguments>

After each Action, the environment will reply with:

    Observation: <the tool's output>

Then you produce the next Thought / Action / Action Input. Repeat until the
task is done. To finish, call the special `final_answer` tool:

    Thought: I have everything I need.
    Action: final_answer
    Action Input: {{"answer": "…your clean final message to the user…"}}

## Rules
- Prefer looking things up (`web_search` + `fetch_url`) over guessing.
- If a package or CLI tool is missing, install it with `run_shell` and continue.
- Uploaded files (if any) will be listed to you at the start of the task.
- If the user gave you an image, you may just describe it — the image is
  already in your vision context. If you want to look at an image you
  produced or downloaded yourself, call `view_image(path)` first.
- Keep Thoughts short. Do not dump code inside a Thought — put code in an
  Action Input for `run_python` or `run_shell`.
- Action Input MUST be a single valid JSON object on ONE logical line
  (multi-line strings are fine, but the outer braces must parse as JSON).
- Do not fabricate Observations. Wait for the real tool result.

## Available tools
{_tool_descriptions_for_prompt()}
"""

# ---------------------------------------------------------------------
# Prompt assembly + LLM call
# ---------------------------------------------------------------------
def _build_messages(history_msgs: List[Dict[str, Any]],
                    user_turn: Dict[str, Any],
                    scratchpad: str) -> List[Dict[str, Any]]:
    """
    Build the qwen-vl chat messages list. Any images uploaded on this
    turn (or previously view_image'd) are attached as vision inputs to
    the current user message so the model can literally see them.
    """
    msgs: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]
    msgs.extend(history_msgs)

    content: List[Dict[str, Any]] = []
    # Images from THIS turn
    for img_path in user_turn.get("images", []):
        content.append({"type": "image", "image": img_path})
    # Images that were view_image'd during the current ReAct loop
    for _tag, img_path in _VIEWED_IMAGES.items():
        content.append({"type": "image", "image": img_path})

    prose = user_turn["text"]
    if user_turn.get("files"):
        listing = "\n".join(f"  • {f}" for f in user_turn["files"])
        prose += f"\n\n[The user attached the following files, saved to disk:\n{listing}\n]"
    if scratchpad:
        prose += "\n\n" + scratchpad

    content.append({"type": "text", "text": prose})
    msgs.append({"role": "user", "content": content})
    return msgs

def _llm_call(messages: List[Dict[str, Any]], stop: List[str]) -> str:
    """One forward pass through the model. Returns the newly generated text."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
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
    out = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]

    # Enforce our own stop strings by truncating.
    for s in stop:
        idx = out.find(s)
        if idx != -1:
            out = out[:idx]
    return out.strip()

# ---------------------------------------------------------------------
# ReAct step parsing
# ---------------------------------------------------------------------
_STEP_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*"
    r"Action:\s*(?P<action>[a-zA-Z_][a-zA-Z0-9_]*)\s*"
    r"Action Input:\s*(?P<input>.+?)\s*(?=Observation:|Thought:|\Z)",
    re.DOTALL,
)

def _parse_step(chunk: str) -> Optional[Tuple[str, str, str]]:
    m = _STEP_RE.search(chunk)
    if not m:
        return None
    thought = m.group("thought").strip()
    action  = m.group("action").strip()
    raw_in  = m.group("input").strip()
    return thought, action, raw_in

def _parse_json_args(raw: str) -> Dict[str, Any]:
    """
    Robust-ish JSON arg parsing. Try strict JSON first; if that fails,
    try to snip the outermost {...} block; finally, treat the raw text
    as the first positional arg's value.
    """
    raw = raw.strip()
    # strip code fences if the model wrapped it
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    # try to grab the first balanced {...}
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end+1])
        except Exception:
            pass
    return {"_raw": raw}

# ---------------------------------------------------------------------
# The core streaming ReAct loop
# ---------------------------------------------------------------------
MAX_STEPS = 12
STOP_TOKENS = ["Observation:"]

def agent_stream(user_text: str,
                 uploaded_paths: List[str],
                 history_msgs: List[Dict[str, Any]],
                 expiry_hours: int
                 ) -> Generator[Tuple[str, str, List[str], List[Dict[str, Any]]], None, None]:
    """
    Yields (thought_log_markdown, final_answer_or_empty, delivered_paths,
    updated_history_msgs) as the agent runs. When final_answer_or_empty
    is non-empty the loop is done.
    """
    global _DELIVERED, _VIEWED_IMAGES, _TEMPFILE_EXPIRY
    _DELIVERED = []
    _VIEWED_IMAGES = {}
    _TEMPFILE_EXPIRY = expiry_hours if expiry_hours in VALID_EXPIRY else 1

    # Sort uploads into images vs everything else — images go into vision.
    image_paths, other_files = [], []
    for p in uploaded_paths:
        mime, _ = mimetypes.guess_type(p)
        if (mime or "").startswith("image/"):
            image_paths.append(p)
        else:
            other_files.append(p)

    user_turn = {
        "text":   user_text,
        "images": image_paths,
        "files":  uploaded_paths,
    }

    log_lines: List[str] = []
    def log(line: str):
        log_lines.append(line)
        return "\n".join(log_lines)

    yield log("▸ Thinking about how to approach this…"), "", [], history_msgs

    scratchpad = ""
    for step_i in range(1, MAX_STEPS + 1):
        messages = _build_messages(history_msgs, user_turn, scratchpad)
        try:
            raw = _llm_call(messages, stop=STOP_TOKENS)
        except Exception as e:
            err = f"[LLM error] {type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}"
            yield log(f"❌ {err}"), err, [], history_msgs
            return

        parsed = _parse_step(raw)
        if not parsed:
            # Model produced free text instead of a proper step — accept
            # it as the final answer and stop.
            final = raw.strip() or "(the model produced no answer)"
            yield log("✔ done"), final, [d["path"] for d in _DELIVERED], history_msgs + [
                {"role": "user",      "content": [{"type": "text", "text": user_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": final}]},
            ]
            return

        thought, action, raw_input = parsed
        args = _parse_json_args(raw_input)

        # Log the thought + intended action to the UI.
        pretty_args = json.dumps(args, ensure_ascii=False)
        if len(pretty_args) > 240:
            pretty_args = pretty_args[:240] + "…"
        yield (log(f"💭 {thought}"), "", [d['path'] for d in _DELIVERED], history_msgs)
        yield (log(f"🔧 Calling tool: `{action}` {pretty_args}"), "",
               [d['path'] for d in _DELIVERED], history_msgs)

        # -------------------- final_answer short-circuit
        if action == "final_answer":
            final = args.get("answer") or args.get("_raw") or ""
            yield (log("✔ done"), final,
                   [d["path"] for d in _DELIVERED],
                   history_msgs + [
                       {"role": "user",      "content": [{"type": "text", "text": user_text}]},
                       {"role": "assistant", "content": [{"type": "text", "text": final}]},
                   ])
            return

        # -------------------- dispatch
        if action not in TOOLS:
            observation = f"[error] unknown tool: {action}"
        else:
            spec = TOOLS[action]
            fn = spec["fn"]
            try:
                call_args = {}
                for name in spec["args"]:
                    if name in args:
                        call_args[name] = args[name]
                # Fallback: if the model dumped a raw string, use it as
                # the first positional arg.
                if not call_args and "_raw" in args and spec["args"]:
                    call_args[spec["args"][0]] = args["_raw"]
                observation = fn(**call_args) if fn else "[error] no impl"
            except Exception as e:
                observation = f"[tool error] {type(e).__name__}: {e}"

        # Log the observation (truncated) to the UI.
        obs_preview = observation.strip().splitlines()
        if obs_preview:
            head = obs_preview[0][:200]
            more = "" if len(obs_preview) <= 1 else f"  (+{len(obs_preview)-1} more lines)"
            yield (log(f"📄 {head}{more}"), "",
                   [d["path"] for d in _DELIVERED], history_msgs)

        # Append this step to the scratchpad for the next LLM call.
        scratchpad += (
            f"\nThought: {thought}\n"
            f"Action: {action}\n"
            f"Action Input: {json.dumps(args, ensure_ascii=False)}\n"
            f"Observation: {observation}\n"
        )

    # Loop exhausted without final_answer.
    final = ("I ran out of reasoning steps before finishing this task. "
             "Here's what I did so far — you may want to ask a narrower "
             "follow-up.\n\n" + scratchpad[-2000:])
    yield (log("⚠ max steps reached"), final,
           [d["path"] for d in _DELIVERED],
           history_msgs + [
               {"role": "user",      "content": [{"type": "text", "text": user_text}]},
               {"role": "assistant", "content": [{"type": "text", "text": final}]},
           ])

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

def _format_delivered_summary(delivered: List[Dict[str, Any]]) -> str:
    if not delivered:
        return ""
    lines = ["", "---", "**📎 Files delivered:**"]
    for d in delivered:
        line = f"- `{d['name']}` ({d['size']:,} bytes)"
        if d.get("description"):
            line += f" — {d['description']}"
        up = d.get("tempfile") or {}
        if up.get("ok"):
            line += f"  \n  🔗 [{up['url']}]({up['url']}) *(expires in {up['expiry_hours']}h)*"
        else:
            line += f"  \n  *(tempfile.org upload skipped: {up.get('error','n/a')} — use the direct download below)*"
        lines.append(line)
    return "\n".join(lines)

def _initial_history() -> List[Dict[str, Any]]:
    return []

# ---------- Gradio callback (a generator, for live streaming) ----------
def on_submit(user_msg: str,
              files,
              expiry_choice: str,
              chat_history: List[Dict[str, str]],
              agent_history: List[Dict[str, Any]]):
    """
    Gradio streaming handler. Yields updates to:
      (chatbot, thought_log, file_output, agent_history_state, uploads_state, input_box)
    """
    user_msg = (user_msg or "").strip()
    uploads = _copy_uploads(files)
    if not uploads:
        uploads = [str(Path(p)) for p in getattr(builtins, "ANVIL_FILES", [])
                   if Path(p).exists()]
    if not user_msg and not uploads:
        yield chat_history, "*(nothing to do)*", [], agent_history, None, ""
        return

    if not user_msg and uploads:
        user_msg = "(no text — please look at the uploaded file(s) and decide what to do.)"

    # Append the user turn to the visible chat immediately.
    display_user = user_msg
    if uploads:
        display_user += "\n\n" + "\n".join(f"📎 `{Path(p).name}`" for p in uploads)
    chat_history = _chatbot_history_add_turn(chat_history, display_user)
    yield chat_history, "▸ Starting…", [], agent_history, None, ""

    expiry = int(expiry_choice.split()[0]) if expiry_choice else 1

    final_answer = ""
    delivered_paths: List[str] = []
    last_log = ""
    new_agent_history = agent_history

    for log_md, final, dpaths, hist in agent_stream(
            user_msg, uploads, agent_history, expiry):
        last_log = log_md
        _chatbot_history_set_assistant(chat_history, final or "*thinking…*")
        yield chat_history, last_log, dpaths, hist, None, ""
        if final:
            final_answer = final
            delivered_paths = dpaths
            new_agent_history = hist

    # Compose the final chat message with a summary of delivered files.
    summary = _format_delivered_summary(_DELIVERED)
    _chatbot_history_set_assistant(chat_history, (final_answer or "(no answer)") + summary)
    yield chat_history, last_log + "\n\n**✅ Finished.**", delivered_paths, new_agent_history, None, ""

# ---------------------------------------------------------------------
# Simple cell UI
# ---------------------------------------------------------------------

def launch():
    with gr.Blocks() as demo:
        gr.Markdown("## Anvil")
        gr.Markdown("Ask a question, attach files if needed, and press Send.")

        agent_hist_state = gr.State([])
        uploads_state = gr.State([])

        chatbot_params = {
            "type": "messages",
            "height": 500,
            "show_label": False,
            "show_copy_button": True,
            "render_markdown": True,
            "value": _chatbot_history_initial(),
        }
        chatbot_signature = inspect.signature(gr.Chatbot.__init__).parameters
        chatbot = gr.Chatbot(**{
            key: value for key, value in chatbot_params.items()
            if key in chatbot_signature
        })

        input_box = gr.Textbox(
            label="Message",
            placeholder="What would you like Anvil to do?",
            lines=3,
        )
        file_input = gr.File(
            label="Attachments",
            file_count="multiple",
        )
        expiry_dd = gr.Dropdown(
            choices=["1 hour", "6 hours", "24 hours", "48 hours"],
            value="1 hour",
            label="Download link expiry",
        )

        with gr.Row():
            send_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear")

        thought_log = gr.Markdown("Idle.", label="Activity")
        file_output = gr.File(label="Downloads", file_count="multiple", interactive=False)

        submit_kwargs = dict(
            fn=on_submit,
            inputs=[input_box, file_input, expiry_dd, chatbot, agent_hist_state],
            outputs=[chatbot, thought_log, file_output, agent_hist_state, file_input, input_box],
        )
        send_btn.click(**submit_kwargs)
        input_box.submit(**submit_kwargs)

        def _clear():
            return _chatbot_history_initial(), "Idle.", [], []

        clear_btn.click(
            _clear,
            outputs=[chatbot, thought_log, file_output, agent_hist_state],
        )

    print(f"Launching Anvil. Model: {MODEL_ID}. GPU: {GPU_NAME}.")
    try:
        queued_demo = demo.queue(default_concurrency_limit=1)
    except TypeError:
        queued_demo = demo.queue()
    queued_demo.launch(
        share=True,
        debug=False,
        inline=True,
        show_error=True,
    )

if __name__ == "__main__":
    launch()
