# Anvil

**A compact, polished four-cell autonomous agent for Google Colab, backed by maintainable Python files.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motionssalt/anvil/blob/main/anvil.ipynb)

Anvil turns a Colab GPU runtime into a persistent, tool-using AI agent with a Gradio chat and file-upload interface. Like the MOTIONSALT Upscaler notebook, the visible notebook is intentionally compact: each cell has a clean Colab form title, implementation details remain hidden behind **Show code**, and the actual agent logic lives in `src/anvil/`.

## The Colab experience

Run the four cells from top to bottom. The notebook presents the workflow as four compact cards:

| Cell | What it does |
| --- | --- |
| **Connect** | Creates the workspace, downloads the repository when needed, installs dependencies, checks the GPU, and loads the model with a polished progress card. |
| **Upload a file** | Provides a compact file picker for images, documents, data, audio, and video. Selected files are made available to the agent. |
| **Chat with Anvil** | Launches the Gradio chat, live thought-process pane, attachment support, and generated-file downloads. |
| **Download** | Gives a clean reminder that generated files are available from the chat’s Downloads panel. |

The notebook itself contains only short wrapper cells. Users can leave every cell collapsed and interact with the rendered cards, exactly as in the reference Upscaler notebook.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/anvil/setup.py` | Dependency installation, GPU checks, model loading, and runtime state. |
| `src/anvil/agent.py` | Tool implementations, ReAct loop, Gradio callbacks, and UI launcher. |
| `scripts/run_anvil.py` | Non-notebook entry point that sets up and launches Anvil. |
| `anvil.ipynb` | Compact four-cell Colab wrapper with form-cell metadata and progress/file-picker UI. |
| `requirements.txt` | Runtime dependencies. |

## Run from a GPU VM

```bash
pip install -e .
python scripts/run_anvil.py
```

The default model is `Qwen/Qwen2.5-VL-7B-Instruct` loaded in 4-bit mode for a free-tier T4. The agent provides Python and shell execution, file I/O, image viewing, web search/fetch, and dual-mode file delivery through the Gradio UI.

The chat launcher detects the installed Gradio `Chatbot` API at runtime. It uses message history when `type="messages"` is supported and automatically falls back to the legacy tuple history format when it is not, preventing the `Chatbot.__init__() got an unexpected keyword argument 'type'` and initial-history format errors seen in some Colab runtimes.

The model loader is tuned for a Colab T4: it reserves VRAM headroom, enables the BitsAndBytes CPU-offload option required for non-quantized modules, and supplies an offload directory plus explicit CPU memory to Accelerate. This avoids the quantized-model dispatch failure that occurs when a 14.6 GB T4 cannot hold every module on the GPU.

## Notes

The first notebook cell clones the repository into `/content/anvil` when the source files are not already present, so the notebook works when opened directly from GitHub in Colab. The implementation is kept outside the notebook so it can be imported, tested, and run from the command line without turning the notebook into a wall of raw source code.
