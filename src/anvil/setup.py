"""Anvil runtime setup and model loading for Google Colab or a GPU VM."""

import builtins
import subprocess
import time
from typing import Any, Dict

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LOAD_IN_4BIT = True
MAX_NEW_TOKENS = 1024
PKGS = [
    "torch", "transformers>=4.45.0", "accelerate>=0.34.0",
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
    _run("pip install -q " + " ".join(f'\"{package}\"' for package in PKGS))

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

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    _log(f"loading model: {MODEL_ID}   (4-bit={LOAD_IN_4BIT})")
    _log("first-time download is ~5–8 GB of weights, please be patient …")
    bnb_cfg = None
    if LOAD_IN_4BIT:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            # Required when Accelerate places any non-quantized modules on CPU.
            llm_int8_enable_fp32_cpu_offload=True,
        )
    started = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Do not use automatic device placement here. On a T4, Accelerate may place one
    # small module on CPU or disk, and BitsAndBytes then rejects the whole
    # quantized load before the model is constructed. The 4-bit 7B model fits
    # on a 14.6 GB T4, so keep the complete quantized model on CUDA instead.
    device_map = {"": 0}
    _log("loading the complete 4-bit model on CUDA device 0 …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    _log(f"model loaded in {time.time() - started:.1f}s")
    return {"model": model, "processor": processor, "model_id": MODEL_ID,
            "gpu_name": gpu_name, "vram_gb": vram_gb,
            "max_new_tokens": MAX_NEW_TOKENS, "loaded_in_4bit": LOAD_IN_4BIT}

def setup_runtime(install: bool = True) -> Dict[str, Any]:
    if install:
        install_dependencies()
    runtime = load_model()
    builtins.ANVIL = runtime
    print("\n✅ Anvil is ready.")
    print(f"   model : {MODEL_ID}  (4-bit={LOAD_IN_4BIT})")
    print(f"   gpu   : {runtime['gpu_name']}  ({runtime['vram_gb']:.1f} GB)")
    print("👉 Now launch the Gradio interface with `python -m anvil.agent` or run the second Colab cell.")
    return runtime

def main() -> Dict[str, Any]:
    return setup_runtime(install=True)

if __name__ == '__main__':
    main()
