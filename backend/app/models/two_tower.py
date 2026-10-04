"""9. Two-Tower semantic search over the enterprise service catalog.

Query tower and document tower are separate residual MLPs on top of a shared frozen Sentence-BERT
backbone. They're trained with an in-batch InfoNCE contrastive loss on (query, item) pairs. Documents
are pre-encoded into an ANN-style index, so a query costs one tower pass plus a dot product.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from ..core import BaseModel, torch_param_stats
from ..datasets import CATALOG
from ..embedder import embed, embedder_params, get_embedder

CKPT = config.CKPT_DIR / "two_tower.pt"


class Tower(nn.Module):
    def __init__(self, d=384, h=512):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, d))
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x):
        return F.normalize(x + self.ff(x), dim=-1)


class TwoTowerSearch(BaseModel):
    key = "two_tower"
    name = "Two-Tower Semantic Search"
    family = "Dual encoder (contrastive)"
    task = "Semantic search and recommendation"
    modality = "text"
    description = "Separate query and item towers on top of Sentence-BERT, fine-tuned with InfoNCE on service-desk queries. Compared live against frozen-embedding and TF-IDF keyword baselines."
    compute = "CPU · towers 0.8M params · trains in <10 s"

    def _item_text(self, it):
        return f"{it[0]}. {it[1]}"

    def _load(self):
        _, ev = get_embedder()
        self.items = [{"id": i, "title": c[0], "description": c[1]} for i, c in enumerate(CATALOG)]
        self.item_base = torch.tensor(embed([self._item_text(c) for c in CATALOG]))
        d = self.item_base.size(1)
        self.m_q, self.m_d = Tower(d), Tower(d)
        if CKPT.exists() and torch.load(CKPT)["n_items"] == len(CATALOG):
            ck = torch.load(CKPT)
            self.m_q.load_state_dict(ck["q"]); self.m_d.load_state_dict(ck["d"]); self.curve = ck["curve"]
        else:
            self.curve = self._train()
        self.m_q.eval(); self.m_d.eval()
        with torch.no_grad():
            self.index = self.m_d(self.item_base)
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([self._item_text(c) for c in CATALOG])
        self.tfidf_mat = self.tfidf.transform([self._item_text(c) for c in CATALOG])
        self.variant = f"{ev} + residual towers"
        bp, bmb = embedder_params()
        s = torch_param_stats(self.m_q, self.m_d)
        self.params, self.param_mb = s["params"] + bp, s["param_mb"] + bmb
        self.tower_params = s["params"]

    def _train(self, epochs=60):
        qs, ys = [], []
        for i, c in enumerate(CATALOG):
            for q in c[2] + [c[0]]:
                qs.append(q); ys.append(i)
        Q = torch.tensor(embed(qs)); Y = torch.tensor(ys)
        opt = torch.optim.AdamW(list(self.m_q.parameters()) + list(self.m_d.parameters()), lr=1e-3, weight_decay=0.01)
        curve = []
        for ep in range(1, epochs + 1):
            perm = torch.randperm(len(Q))
            tot = 0.0
            for b in range(0, len(Q), 32):
                ix = perm[b:b + 32]
                qv = self.m_q(Q[ix])
                dv = self.m_d(self.item_base)  # score against the full catalog (all negatives)
                loss = F.cross_entropy(qv @ dv.T / 0.05, Y[ix])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(ix)
            if ep % 6 == 0:
                curve.append((ep, round(tot / len(Q), 4)))
        torch.save({"q": self.m_q.state_dict(), "d": self.m_d.state_dict(), "curve": curve, "n_items": len(CATALOG)}, CKPT)
        return curve

    def _rank(self, query, method):
        if method == "tfidf":
            s = (self.tfidf_mat @ self.tfidf.transform([query]).T).toarray().ravel()
        else:
            q = torch.tensor(embed([query]))
            with torch.no_grad():
                s = (self.m_q(q) @ self.index.T if method == "two_tower" else q @ self.item_base.T)[0].numpy()
        return s

    def search(self, query: str, k: int = 5):
        out = {}
        for method in ("two_tower", "frozen_sbert", "tfidf"):
            t0 = time.perf_counter()
            s = self._rank(query, method)
            ms = (time.perf_counter() - t0) * 1000
            order = np.argsort(-s)[:k]
            out[method] = {"ms": round(ms, 2), "results": [dict(self.items[i], score=round(float(s[i]), 4)) for i in order]}
        return {"query": query, "methods": out, "training_curve": self.curve}

    def benchmark(self):
        res = {}
        for method in ("two_tower", "frozen_sbert", "tfidf"):
            r1 = r3 = mrr = 0; n = 0; lat = []
            for i, c in enumerate(CATALOG):
                for q in c[3]:
                    t0 = time.perf_counter()
                    s = self._rank(q, method)
                    lat.append((time.perf_counter() - t0) * 1000)
                    rank = int((s > s[i]).sum()) + 1
                    r1 += rank == 1; r3 += rank <= 3; mrr += 1 / rank; n += 1
            res[method] = {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3), "MRR": round(mrr / n, 3),
                           "ms": round(sum(lat) / n, 2)}
        return {"primary": {"name": "Recall@3 (held-out)", "value": res["two_tower"]["recall@3"], "better": "higher"},
                "metrics": {"MRR": res["two_tower"]["MRR"], "recall@1": res["two_tower"]["recall@1"],
                            "baseline_sbert_R@3": res["frozen_sbert"]["recall@3"], "baseline_tfidf_R@3": res["tfidf"]["recall@3"]},
                "comparison": res}
