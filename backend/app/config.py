"""Global configuration. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Hugging Face model ids (primary + lightweight fallback)
LLM_MODEL = _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # or meta-llama/Llama-3.2-1B-Instruct (gated)
LLM_FALLBACK = _env("LLM_FALLBACK", "Qwen/Qwen2.5-0.5B-Instruct")
LORA_ADAPTER_DIR = Path(_env("LORA_ADAPTER_DIR", str(CKPT_DIR / "lora_adapter")))

EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

WHISPER_MODEL = _env("WHISPER_MODEL", "openai/whisper-base")
WHISPER_FALLBACK = _env("WHISPER_FALLBACK", "openai/whisper-tiny")

YOLO_WEIGHTS = _env("YOLO_WEIGHTS", "yolov8n.pt")

BERT_MODEL = _env("BERT_MODEL", "nlptown/bert-base-multilingual-uncased-sentiment")
BERT_FALLBACK = _env("BERT_FALLBACK", "distilbert-base-uncased-finetuned-sst-2-english")

CLIP_MODEL = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")

# Keep at most this many heavy models resident at once (8 GB Macs need this).
MAX_RESIDENT_HEAVY = int(_env("MAX_RESIDENT_HEAVY", "3"))


def pick_device() -> str:
    forced = os.environ.get("DEVICE")
    if forced:
        return forced
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch missing
        pass
    return "cpu"


DEVICE = pick_device()
