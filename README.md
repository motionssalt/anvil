# Anvil

**A two-cell autonomous agent notebook for Google Colab.**

Anvil turns a free (or paid) Colab GPU runtime into a persistent,
tool-using AI agent with a real chat + file-upload interface, running
entirely inside the notebook. Open it, run two cells, and you have a
working agent — no external server, no local install, works from a
phone browser.

---

## What it is

Anvil is a single notebook (`anvil.ipynb`) with exactly two cells:

1. **Cell 1 — Setup.** Installs dependencies, verifies the GPU,
   downloads and loads a vision-capable, tool-calling LLM (default:
   `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit-quantized to fit a free T4)
   and prints clear progress the whole way.

2. **Cell 2 — The agent interface.** Launches a Gradio app (with
   `share=True` so you get a public URL that works reliably on
   mobile) which IS the entire product from that point on. Chat +
   file upload + live "thought process" pane + dual-delivery file
   output.

Once cell 2 is running you do not touch the notebook again until
you're done.

## The agent

Anvil is not a chatbot answering from memory — it is a genuine
tool-using agent with a ReAct-style reasoning loop and the following
tools:

- **`run_python`** — execute arbitrary Python inside the Colab VM.
- **`run_shell`** — execute arbitrary shell commands (including
  `pip install …` and `apt-get install -y …`) so it can install its
  own dependencies mid-task if a job needs `ffmpeg`, `yt-dlp`,
  `pandas`, whatever.
- **`read_file`** / **`write_file`** — file I/O against the Colab VM.
- **`view_image`** — feed an uploaded image straight into the
  vision-capable model as part of reasoning (no separate captioning
  step).
- **`web_search`** + **`fetch_url`** — DuckDuckGo search and page
  fetching so the agent can look things up rather than guess.
- **`deliver_file`** — hands a produced file to the user in TWO ways
  at once: a direct Gradio download component **and** a temporary
  hosted link via [tempfile.org](https://tempfile.org/api) (expiry
  configurable in the UI — 1 / 6 / 24 / 48 hours, default 1h).

The agent can chain many tool calls across many reasoning steps to
complete a single request.

## Live thought-process streaming

While the agent is working you get a continuously-updating pane
showing what it is doing right now:

```
▸ Thinking about how to approach this…
▸ Calling tool: web_search("…")
▸ Installing ffmpeg…
▸ Reading uploaded file: screenshot.png
▸ Running code: …
▸ ✔ done
```

The final clean answer is delivered separately as the actual chat
reply — the log is process, the message is the answer.

## Requirements

- Google Colab with a **GPU runtime**. A free-tier **T4 (16 GB VRAM)**
  is enough for the default 4-bit `Qwen2.5-VL-7B-Instruct`. An A100 /
  L4 will obviously run it faster and let you pick a larger model in
  the setup cell.
- No API keys required for the default configuration.

## How to run

1. Open `anvil.ipynb` in Google Colab
   (`File → Upload notebook`, or push this repo to GitHub and open it
   via `Open notebook → GitHub`).
2. `Runtime → Change runtime type → GPU` (T4 is fine).
3. Run **Cell 1**. Wait for the "✅ Anvil is ready" line — first run
   takes a few minutes to download the model weights.
4. Run **Cell 2**. Two URLs will print:
   - a local Colab iframe URL,
   - a public `*.gradio.live` URL (use this one on your phone).
5. Chat with the agent. Upload files with the paperclip. Watch the
   live thought-process pane on the right.

To stop it, interrupt Cell 2. To start again, just re-run Cell 2 —
you don't need to reload the model.

## Example prompts to try

- *"Download the latest xkcd comic, describe what's happening in the
  picture, and give it back to me as a file."*
  → exercises `web_search` + `fetch_url` + `run_shell` + `view_image`
  + `deliver_file`.

- *"Here's a CSV [attach file]. Figure out what's in it, make a
  reasonable chart, and hand me back the PNG."*
  → exercises `read_file` + `run_python` (self-installing `pandas` /
  `matplotlib` if needed) + `deliver_file`.

- *"What's the current top story on Hacker News right now, and does
  the linked article actually say what the headline claims?"*
  → exercises `web_search` + `fetch_url` twice + reasoning.

- *"Look at this screenshot [attach image] and write me a Python
  script that reproduces the layout in matplotlib."*
  → exercises `view_image` + `run_python` + `deliver_file`.

- *"Grab any short public-domain audio clip, transcribe it, and give
  me both the audio and the transcript."*
  → exercises self-installing `ffmpeg` + `openai-whisper` (or similar)
  mid-task, plus `deliver_file` twice.

## Notes on the environment

- Colab sessions are **not permanent** — when the runtime is
  recycled, anything written to `/content` is gone. Save anything
  you want to keep via the tempfile.org link or the direct download
  while the session is live.
- The agent knows it's in a Colab VM on a T4 (this is baked into its
  system prompt) and will be mindful of VRAM in code it runs.
- Files larger than 100 MB skip the tempfile.org upload
  automatically and fall back to Gradio-only download (per the
  tempfile.org documented limit).

## Project name

**Anvil** — because it's the flat, solid surface you hammer things
into shape on.
