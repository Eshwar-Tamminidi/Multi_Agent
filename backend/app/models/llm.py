"""3. Small instruction-tuned LLM (Qwen 2.5 1.5B / Llama 3.2 1B) with optional LoRA adapter."""
import threading
import time

import torch

from .. import config
from ..core import BaseModel, torch_param_stats
from ..datasets import RAG_QA

SYSTEM_PROMPT = "You are Nimbus Assistant, a concise and helpful enterprise AI assistant."


class SmallLLM(BaseModel):
    key = "llm"
    name = "Qwen 2.5 / Llama 3.2 + LoRA"
    family = "Decoder-only LLM (+ PEFT LoRA)"
    task = "Chat, instruction following and RAG answers"
    modality = "text"
    description = "A 1-1.5B-parameter instruction-tuned LLM in fp16. If a LoRA adapter is found (see scripts/train_lora.py), it's attached through PEFT and can be switched on and off for A/B comparison."
    heavy = True
    compute = "MPS/CUDA recommended · ~3 GB fp16"

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if config.DEVICE in ("cuda", "mps") else torch.float32
        last_err = None
        for mid in (config.LLM_MODEL, config.LLM_FALLBACK):
            try:
                tok = AutoTokenizer.from_pretrained(mid)
                model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, low_cpu_mem_usage=True)
                self.variant = mid.split("/")[-1]
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.notes.append(f"Couldn't load {mid}: {type(e).__name__}")
        else:
            raise RuntimeError(f"No LLM could be loaded: {last_err}")

        self.has_lora = False
        if config.LORA_ADAPTER_DIR.exists():
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(config.LORA_ADAPTER_DIR))
                self.has_lora = True
                self.variant += " + LoRA"
            except Exception as e:  # noqa: BLE001
                self.notes.append(f"LoRA adapter not attached: {e}")
        self.m_model = model.to(config.DEVICE).eval()
        self.m_tok = tok
        s = torch_param_stats(self.m_model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    def _inputs(self, messages):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        ids = self.m_tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        return ids.to(config.DEVICE)

    def _gen_kwargs(self, max_new_tokens, temperature):
        kw = dict(max_new_tokens=int(max_new_tokens), pad_token_id=self.m_tok.eos_token_id)
        if temperature and temperature > 0.01:
            kw.update(do_sample=True, temperature=float(temperature), top_p=0.9)
        else:
            kw.update(do_sample=False)
        return kw

    def _ctx(self, use_lora):
        if self.has_lora and not use_lora:
            return self.m_model.disable_adapter()
        import contextlib

        return contextlib.nullcontext()

    @torch.no_grad()
    def chat(self, messages, max_new_tokens=256, temperature=0.7, use_lora=True):
        ids = self._inputs(messages)
        t0 = time.perf_counter()
        with self._ctx(use_lora):
            out = self.m_model.generate(ids, attention_mask=torch.ones_like(ids), **self._gen_kwargs(max_new_tokens, temperature))
        dt = time.perf_counter() - t0
        new = out[0, ids.shape[1]:]
        return {"reply": self.m_tok.decode(new, skip_special_tokens=True).strip(),
                "prompt_tokens": int(ids.shape[1]), "new_tokens": int(new.shape[0]),
                "tokens_per_sec": round(new.shape[0] / dt, 2), "lora_active": self.has_lora and use_lora}

    def stream(self, messages, max_new_tokens=256, temperature=0.7, use_lora=True):
        """Yields text deltas, then a final stats dict."""
        from transformers import TextIteratorStreamer

        ids = self._inputs(messages)
        streamer = TextIteratorStreamer(self.m_tok, skip_prompt=True, skip_special_tokens=True)
        kw = self._gen_kwargs(max_new_tokens, temperature)

        def run():
            with torch.no_grad(), self._ctx(use_lora):
                self.m_model.generate(ids, attention_mask=torch.ones_like(ids), streamer=streamer, **kw)

        t0 = time.perf_counter()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        ttft, n, text = None, 0, ""
        for piece in streamer:
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                text += piece
                yield piece
        th.join()
        dt = time.perf_counter() - t0
        ntok = len(self.m_tok(text)["input_ids"]) if text else 0
        yield {"done": True, "ttft_ms": round((ttft or dt) * 1000, 1), "total_ms": round(dt * 1000, 1),
               "new_tokens": ntok, "tokens_per_sec": round(ntok / dt, 2) if dt else None,
               "variant": self.variant, "lora_active": self.has_lora and use_lora}

    def benchmark(self):
        kb = {p.stem: p.read_text() for p in (config.DATA_DIR / "knowledge").glob("*.md")}
        ok, tps, lat = 0, [], []
        subset = RAG_QA[::3]
        for q, kws, src in subset:
            msg = [{"role": "user", "content": f"Context:\n{kb[src]}\n\nAnswer in one sentence: {q}"}]
            t0 = time.perf_counter()
            r = self.chat(msg, max_new_tokens=48, temperature=0)
            lat.append((time.perf_counter() - t0) * 1000)
            tps.append(r["tokens_per_sec"])
            ok += any(k.lower() in r["reply"].lower() for k in kws)
        n = len(subset)
        return {
            "primary": {"name": "Grounded QA accuracy", "value": round(ok / n, 3), "better": "higher"},
            "metrics": {"avg_tokens_per_sec": round(sum(tps) / n, 2), "avg_response_ms": round(sum(lat) / n, 1),
                        "questions": n, "lora": self.has_lora},
        }
