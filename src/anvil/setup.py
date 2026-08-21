"""Anvil runtime setup and model loading for Google Colab or a GPU VM."""

import builtins
import subprocess
import time
from typing import Any, Dict

# Qwen3-VL-8B-Instruct: vision-capable with *native* structured tool calling
# (its chat template emits <tool_call>{...}</tool_call> blocks). This replaces
# Qwen2.5-VL-7B-Instruct, whose 4-bit build was too unreliable for multi-step
# agentic tool use (fabricated observations, skipped actions, early give-up).
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 2048

# VRAM heuristic: Qwen3-VL-8B needs ~19 GB in fp16 (weights + activations +
# KV cache). Only GPUs with comfortable headroom load full precision;
# everything else (T4 15 GB, L4 24 GB) loads NF4 4-bit, which fits easily.
FULL_PRECISION_MIN_VRAM_GB = 30.0

PKGS = [
    "torch", "transformers>=4.57.0", "accelerate>=0.34.0",
    "bitsandbytes>=0.43.0", "sentencepiece", "safetensors", "pillow",
    "qwen-vl-utils", "gradio>=4.44.0", "duckduckgo-search>=6.2.0",
    "beautifulsoup4", "lxml", "readability-lxml", "requests", "httpx",
    "markdownify",
]


def _log(message: str) -> None:
    print(f"[anvil-setup] {message}", flush=True)


def _run(command: str, check: bool = True):
    _log(f"$ {command}")
    result = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-6:])
    if tail:
        print(tail, flush=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")
    return result


def install_dependencies() -> None:
    _log("installing Python packages (this takes ~1–2 minutes the first time) …")
    _run("pip install -q --upgrade pip")
    _run("pip install -q " + " ".join(f'"{package}"' for package in PKGS))


def load_model() -> Dict[str, Any]:
    _log("checking GPU …")
    try:
        _run("nvidia-smi -L")
    except Exception:
        print("\n❌ No GPU detected. In Colab: Runtime → Change runtime type → GPU.\n")
        raise

    import torch
    _log(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available — restart the runtime with GPU selected.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    _log(f"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

    load_in_4bit = vram_gb < FULL_PRECISION_MIN_VRAM_GB
    # bf16 only where the hardware supports it (Ampere+); fp16 otherwise.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    _log(f"loading model: {MODEL_ID}   (4-bit={load_in_4bit})")
    _log("first-time download is ~17 GB of weights, please be patient …")

    bnb_cfg = None
    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            # Required when Accelerate places any non-quantized modules on CPU.
            llm_int8_enable_fp32_cpu_offload=True,
        )

    started = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Keep the complete model on CUDA: on small GPUs Accelerate may otherwise
    # spill a module to CPU/disk, which BitsAndBytes rejects mid-load.
    device_map = {"": 0}
    _log(f"loading the complete model on CUDA device 0 "
         f"({'4-bit NF4' if load_in_4bit else str(dtype).replace('torch.', '')}) …")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    _log(f"model loaded in {time.time() - started:.1f}s")
    return {"model": model, "processor": processor, "model_id": MODEL_ID,
            "gpu_name": gpu_name, "vram_gb": vram_gb,
            "max_new_tokens": MAX_NEW_TOKENS, "loaded_in_4bit": load_in_4bit}


def setup_runtime(install: bool = True) -> Dict[str, Any]:
    if install:
        install_dependencies()
    runtime = load_model()
    builtins.ANVIL = runtime
    print("\n✅ Anvil is ready.")
    print(f"   model : {MODEL_ID}  (4-bit={runtime['loaded_in_4bit']})")
    print(f"   gpu   : {runtime['gpu_name']}  ({runtime['vram_gb']:.1f} GB)")
    print("👉 Now launch the Gradio interface with `python -m anvil.agent` or run the second Colab cell.")
    return runtime


def main() -> Dict[str, Any]:
    return setup_runtime(install=True)


if __name__ == '__main__':
    main()
