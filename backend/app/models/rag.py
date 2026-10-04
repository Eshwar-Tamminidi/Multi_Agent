"""2. Retrieval-Augmented Generation over the enterprise knowledge base using Sentence-BERT embeddings."""
import re
import time

import numpy as np

from .. import config
from ..core import BaseModel, registry
from ..datasets import RAG_QA
from ..embedder import embed, embedder_params, get_embedder


def chunk_markdown(stem: str, text: str):
    title = stem
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
    chunks = []
    for sec in re.split(r"^##\s+", text, flags=re.M)[1:] or [text]:
        lines = sec.strip().split("\n", 1)
        heading = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else heading
        chunks.append({"doc": stem, "title": title, "section": heading, "text": body})
    return chunks


def split_sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


class RAGSystem(BaseModel):
    key = "rag"
    name = "RAG + Sentence-BERT"
    family = "Bi-encoder retriever + generator"
    task = "Enterprise knowledge retrieval"
    modality = "text"
    description = "Splits company documents into chunks, embeds them with all-MiniLM-L6-v2 and finds the closest chunks by cosine similarity. Answers come from the LLM when it's loaded, or are picked straight from the retrieved sentences."
    compute = "CPU · 22M-param encoder · ~90 MB"

    def _load(self):
        model, variant = get_embedder()
        self.variant = variant
        self.params, self.param_mb = embedder_params()
        self.chunks = []
        for p in sorted((config.DATA_DIR / "knowledge").glob("*.md")):
            self.chunks += chunk_markdown(p.stem, p.read_text(encoding="utf-8"))
        self._reindex()

    def _reindex(self):
        texts = [f"{c['title']} - {c['section']}: {c['text']}" for c in self.chunks]
        self.matrix = embed(texts)

    def add_document(self, name: str, text: str):
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.rsplit(".", 1)[0])[:60] or "upload"
        new = chunk_markdown(stem, text if text.lstrip().startswith("#") else f"# {name}\n\n## Content\n{text}")
        # If no headings, split long text into ~600 char windows
        if len(new) == 1 and len(new[0]["text"]) > 900:
            body = new[0]["text"]
            new = [{"doc": stem, "title": name, "section": f"Part {i + 1}", "text": body[j:j + 700]}
                   for i, j in enumerate(range(0, len(body), 600))]
        self.chunks += new
        self._reindex()
        return {"added_chunks": len(new), "total_chunks": len(self.chunks)}

    def documents(self):
        docs = {}
        for c in self.chunks:
            docs.setdefault(c["doc"], {"doc": c["doc"], "title": c["title"], "sections": 0})
            docs[c["doc"]]["sections"] += 1
        return list(docs.values())

    def retrieve(self, query: str, k: int = 3):
        t0 = time.perf_counter()
        q = embed([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        ms = (time.perf_counter() - t0) * 1000
        return [dict(self.chunks[i], score=float(sims[i])) for i in order], q, ms

    def extractive_answer(self, query, q_vec, hits):
        sents = []
        for h in hits[:2]:
            sents += split_sentences(h["text"])
        if not sents:
            return ""
        sv = embed(sents)
        best = np.argsort(-(sv @ q_vec))[:2]
        return " ".join(sents[i] for i in sorted(best))

    def answer(self, query: str, k: int = 3, use_llm: bool = False):
        hits, q_vec, retr_ms = self.retrieve(query, k)
        mode = "extractive"
        gen_ms = None
        if use_llm:
            context = "\n\n".join(f"[{i + 1}] ({h['title']} / {h['section']}) {h['text']}" for i, h in enumerate(hits))
            prompt = ("Answer the question using only the context below. Be concise and cite sources like [1].\n\n"
                      f"Context:\n{context}\n\nQuestion: {query}")
            t0 = time.perf_counter()
            llm = registry.ensure_loaded("llm")
            with llm.lock:
                ans = llm.chat([{"role": "user", "content": prompt}], max_new_tokens=160, temperature=0.2)["reply"]
            gen_ms = round((time.perf_counter() - t0) * 1000, 1)
            mode = f"generative ({llm.variant})"
        else:
            ans = self.extractive_answer(query, q_vec, hits)
        return {"answer": ans, "mode": mode, "sources": hits, "retrieval_ms": round(retr_ms, 2), "generation_ms": gen_ms}

    def benchmark(self):
        h1 = h3 = mrr = ans_ok = 0
        lat = []
        for q, kws, src in RAG_QA:
            hits, q_vec, ms = self.retrieve(q, 5)
            lat.append(ms)
            docs = [h["doc"] for h in hits]
            if src in docs:
                r = docs.index(src) + 1
                mrr += 1 / r
                h1 += r == 1
                h3 += r <= 3
            a = self.extractive_answer(q, q_vec, hits[:3]).lower()
            ans_ok += any(k.lower() in a for k in kws)
        n = len(RAG_QA)
        return {
            "primary": {"name": "Answer accuracy", "value": round(ans_ok / n, 3), "better": "higher"},
            "metrics": {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "MRR": round(mrr / n, 3),
                        "avg_retrieval_ms": round(sum(lat) / n, 2), "questions": n, "chunks": len(self.chunks)},
        }
