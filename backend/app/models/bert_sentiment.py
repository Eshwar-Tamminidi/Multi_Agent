"""6. BERT sentiment classification with word-level occlusion explanations."""
import time

import torch

from .. import config
from ..core import BaseModel, torch_param_stats
from ..datasets import SENTIMENT_SET

LABELS = ["negative", "neutral", "positive"]


class BERTSentiment(BaseModel):
    key = "bert"
    name = "BERT Sentiment"
    family = "Encoder-only Transformer"
    task = "Sentiment classification"
    modality = "text"
    description = "A fine-tuned BERT classifier for customer feedback. Gives 3-class sentiment (plus star ratings when available) and highlights which words drove the prediction, found by removing words one at a time."
    heavy = True
    compute = "CPU/MPS · 167M params · ~670 MB"

    def _load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        last = None
        for mid in (config.BERT_MODEL, config.BERT_FALLBACK):
            try:
                self.m_tok = AutoTokenizer.from_pretrained(mid)
                self.m_model = AutoModelForSequenceClassification.from_pretrained(mid).to(config.DEVICE).eval()
                self.variant = mid.split("/")[-1]
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"BERT unavailable: {last}")
        self.n_labels = self.m_model.config.num_labels
        s = torch_param_stats(self.m_model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    @torch.no_grad()
    def _probs3(self, texts):
        enc = self.m_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(config.DEVICE)
        p = torch.softmax(self.m_model(**enc).logits.float(), -1).cpu()
        if self.n_labels == 5:  # 1..5 stars
            stars = p
            p3 = torch.stack([p[:, 0] + p[:, 1], p[:, 2], p[:, 3] + p[:, 4]], 1)
        else:  # binary neg/pos -> derive neutral from uncertainty
            stars = None
            margin = (p[:, 1] - p[:, 0]).abs()
            neu = (1 - margin).clamp(0, 1) ** 2
            p3 = torch.stack([p[:, 0] * (1 - neu), neu, p[:, 1] * (1 - neu)], 1)
            p3 = p3 / p3.sum(1, keepdim=True)
        return p3, stars

    def classify(self, text: str, explain: bool = True):
        p3, stars = self._probs3([text])
        p = p3[0]
        idx = int(p.argmax())
        res = {"label": LABELS[idx], "confidence": round(float(p[idx]), 4),
               "probs": {l: round(float(v), 4) for l, v in zip(LABELS, p)}}
        if stars is not None:
            s = stars[0]
            res["stars"] = round(float((s * torch.arange(1, 6)).sum()), 2)
        if explain:
            words = text.split()[:48]
            if 1 < len(words):
                variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
                pv, _ = self._probs3(variants)
                score = lambda q: float(q[2] - q[0])  # noqa: E731  polarity
                base = score(p)
                res["tokens"] = [{"word": w, "impact": round(base - score(pv[i]), 4)} for i, w in enumerate(words)]
        return res

    def classify_batch(self, texts):
        p3, _ = self._probs3(texts)
        return [{"text": t, "label": LABELS[int(p.argmax())], "confidence": round(float(p.max()), 4)} for t, p in zip(texts, p3)]

    def benchmark(self):
        texts = [t for t, _ in SENTIMENT_SET]
        gold = [y for _, y in SENTIMENT_SET]
        t0 = time.perf_counter()
        p3, _ = self._probs3(texts)
        ms = (time.perf_counter() - t0) * 1000
        pred = p3.argmax(1).tolist()
        acc = sum(int(a == b) for a, b in zip(pred, gold)) / len(gold)
        f1s = []
        for c in range(3):
            tp = sum(1 for a, b in zip(pred, gold) if a == c and b == c)
            fp = sum(1 for a, b in zip(pred, gold) if a == c and b != c)
            fn = sum(1 for a, b in zip(pred, gold) if a != c and b == c)
            pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
            f1s.append(2 * pr * rc / max(1e-9, pr + rc))
        conf = [[sum(1 for a, b in zip(pred, gold) if b == i and a == j) for j in range(3)] for i in range(3)]
        return {"primary": {"name": "Accuracy", "value": round(acc, 3), "better": "higher"},
                "metrics": {"macro_f1": round(sum(f1s) / 3, 3), "batch_ms": round(ms, 1),
                            "ms_per_sample": round(ms / len(texts), 2), "samples": len(texts),
                            "confusion": conf}}
