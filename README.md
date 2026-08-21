# Anvil

**A two-cell autonomous agent for Google Colab, with its implementation in reusable Python files.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/motionssalt/anvil/blob/main/anvil.ipynb)

Anvil turns a Colab GPU runtime into a persistent, tool-using AI agent with a Gradio chat and file-upload interface. The notebook is intentionally a thin launcher: the actual setup and agent implementation live under `src/anvil/`, so they can be imported, tested, run from the command line, and maintained without editing a giant notebook cell.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/anvil/setup.py` | Dependency installation, GPU checks, model loading, and runtime state. |
| `src/anvil/agent.py` | Tool implementations, ReAct loop, Gradio callbacks, and UI launcher. |
| `scripts/run_anvil.py` | Non-notebook entry point that sets up and launches Anvil. |
| `anvil.ipynb` | Small Colab wrapper that imports the Python modules. |
| `requirements.txt` | Runtime dependencies. |

## Run in Colab

Open the notebook, select a GPU runtime, and run the two cells from top to bottom. The first cell installs the local package and loads the model; the second launches the Gradio interface.

## Run from a GPU VM

```bash
pip install -e .
python scripts/run_anvil.py
```

The default model is `Qwen/Qwen2.5-VL-7B-Instruct` loaded in 4-bit mode for a free-tier T4. The agent provides Python and shell execution, file I/O, image viewing, web search/fetch, and dual-mode file delivery through the Gradio UI.
