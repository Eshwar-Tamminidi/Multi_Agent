"""1. Character-level GPT trained from scratch."""
import os
import urllib.request

import torch

from .. import config
from ..core import BaseModel, torch_param_stats
from .tiny_gpt import TinyGPT, evaluate_lm, train_lm

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CKPT = config.CKPT_DIR / "char_gpt.pt"
CORPUS = config.DATA_DIR / "char_corpus.txt"


def load_corpus() -> str:
    if CORPUS.exists():
        return CORPUS.read_text(encoding="utf-8")
    try:
        text = urllib.request.urlopen(SHAKESPEARE_URL, timeout=15).read().decode("utf-8")
    except Exception:
        # Offline fallback: train on the bundled enterprise knowledge base.
        text = "\n\n".join(p.read_text() for p in sorted((config.DATA_DIR / "knowledge").glob("*.md"))) * 20
    CORPUS.write_text(text, encoding="utf-8")
    return text


def train_and_save(steps: int = None, log=print):
    steps = steps or int(os.environ.get("CHAR_GPT_STEPS", "1500"))
    text = load_corpus()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    model = TinyGPT(len(chars), block_size=128, n_layer=4, n_head=4, n_embd=192, dropout=0.1)
    curve = train_lm(model, data[:n], steps, batch_size=48, lr=2e-3, device=config.DEVICE, log=log, val_data=data[n:])
    torch.save({"state": model.cpu().state_dict(), "cfg": model.cfg, "chars": chars, "curve": curve}, CKPT)
    return curve


class CharGPT(BaseModel):
    key = "char_gpt"
    name = "Character-Level GPT"
    family = "Decoder-only Transformer"
    task = "Text generation"
    modality = "text"
    description = "A ~1.8M-parameter GPT trained from scratch to predict the next character. Shows how language models learn spelling, words and style from raw characters."
    compute = "CPU/MPS · trains in ~1-2 min"

    def _load(self):
        if not CKPT.exists():
            self.notes = ["No checkpoint found, so it trained from scratch on first load."]
            train_and_save()
        ck = torch.load(CKPT, map_location="cpu")
        self.m_model = TinyGPT(**ck["cfg"])
        self.m_model.load_state_dict(ck["state"])
        self.m_model.to(config.DEVICE).eval()
        self.chars = ck["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.curve = ck.get("curve", [])
        self.variant = "tinyshakespeare" if len(self.chars) > 60 else "knowledge-base corpus"
        s = torch_param_stats(self.m_model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    def generate(self, prompt: str, max_new_tokens=300, temperature=0.8, top_k=40):
        prompt = "".join(c for c in (prompt or "\n") if c in self.stoi) or "\n"
        idx = torch.tensor([[self.stoi[c] for c in prompt]], device=config.DEVICE)
        out = self.m_model.generate(idx, int(max_new_tokens), float(temperature), int(top_k) if top_k else None)
        text = "".join(self.chars[i] for i in out[0].tolist())
        return {"prompt": prompt, "completion": text[len(prompt):], "tokens": int(max_new_tokens),
                "training_curve": self.curve}

    def benchmark(self):
        text = load_corpus()
        data = torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)
        val = data[int(0.9 * len(data)):]
        loss = evaluate_lm(self.m_model, val, 32, config.DEVICE, iters=20)
        import math
        import time
        t0 = time.perf_counter()
        self.generate("ROMEO:", 200)
        dt = time.perf_counter() - t0
        return {
            "primary": {"name": "Val perplexity", "value": round(math.exp(loss), 3), "better": "lower"},
            "metrics": {"val_loss": loss, "bits_per_char": round(loss / math.log(2), 3),
                        "tokens_per_sec": round(200 / dt, 1)},
        }
