"""FastAPI entry point for the Intelligent Multi-Modal Enterprise AI System."""
import json
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as Schema

from . import config
from .core import registry, rss_mb
from .models.bert_sentiment import BERTSentiment
from .models.char_gpt import CharGPT
from .models.clip_caption import CLIPCaptioner
from .models.llm import SmallLLM
from .models.lstm_ae import LSTMAnomaly
from .models.midi_transformer import MidiTransformer
from .models.rag import RAGSystem
from .models.two_tower import TwoTowerSearch
from .models.whisper_asr import WhisperASR
from .models.yolo import YOLODetector

for cls in (CharGPT, RAGSystem, SmallLLM, WhisperASR, YOLODetector, BERTSentiment, CLIPCaptioner,
            LSTMAnomaly, TwoTowerSearch, MidiTransformer):
    registry.register(cls())

BENCH_FILE = config.CKPT_DIR / "benchmarks.json"
if BENCH_FILE.exists():
    try:
        for k, v in json.loads(BENCH_FILE.read_text()).items():
            if k in registry.models:
                registry.models[k].benchmark_result = v
    except Exception:  # noqa: BLE001
        pass

app = FastAPI(title="Intelligent Multi-Modal Enterprise AI System", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _guard(fn):
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ system / registry
@app.get("/api/system")
def system():
    vm = psutil.virtual_memory()
    return {"device": config.DEVICE, "platform": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(), "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(), "ram_total_gb": round(vm.total / 1e9, 2),
            "ram_used_gb": round(vm.used / 1e9, 2), "ram_percent": vm.percent, "process_rss_mb": round(rss_mb(), 1),
            "loaded": [k for k, m in registry.models.items() if m.status == "ready"],
            "max_resident_heavy": config.MAX_RESIDENT_HEAVY}


@app.get("/api/models")
def models():
    return [m.info() for m in registry.models.values()]


@app.get("/api/models/{key}")
def model_info(key: str):
    return _guard(lambda: registry.get(key).info())


@app.post("/api/models/{key}/load")
def load(key: str):
    return _guard(lambda: registry.ensure_loaded(key).info())


@app.post("/api/models/{key}/unload")
def unload(key: str):
    def f():
        registry.unload(key)
        return registry.get(key).info()
    return _guard(f)


@app.post("/api/models/{key}/benchmark")
def benchmark(key: str):
    def f():
        m = registry.ensure_loaded(key)
        with m.lock:
            t0 = time.perf_counter()
            res = m.benchmark()
            res.update({"benchmark_seconds": round(time.perf_counter() - t0, 2), "timestamp": time.time(),
                        "params": m.params, "param_mb": round(m.param_mb, 1), "load_seconds": m.load_seconds,
                        "load_rss_delta_mb": m.load_rss_delta_mb, "variant": m.variant, "device": config.DEVICE,
                        "avg_inference_ms": m.info()["avg_ms"], "rss_mb": round(rss_mb(), 1)})
            m.benchmark_result = res
        allres = {k: v.benchmark_result for k, v in registry.models.items() if v.benchmark_result}
        BENCH_FILE.write_text(json.dumps(allres, indent=1, default=str))
        return res
    return _guard(f)


@app.get("/api/benchmarks")
def benchmarks():
    return {k: {"name": m.name, "family": m.family, "modality": m.modality, "result": m.benchmark_result,
                "status": m.status, "compute": m.compute} for k, m in registry.models.items()}


# ------------------------------------------------------------------ 1. char-gpt
class GenReq(Schema):
    prompt: str = "ROMEO:"
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 40


@app.post("/api/char_gpt/generate")
def char_gpt(r: GenReq):
    n = max(1, min(r.max_new_tokens, 2000))
    return _guard(lambda: registry.run("char_gpt", lambda m: m.generate(r.prompt, n, r.temperature, r.top_k)))


# ------------------------------------------------------------------ 2. RAG
class AskReq(Schema):
    query: str
    k: int = 3
    use_llm: bool = False


@app.post("/api/rag/ask")
def rag_ask(r: AskReq):
    return _guard(lambda: registry.run("rag", lambda m: m.answer(r.query, r.k, r.use_llm)))


@app.get("/api/rag/documents")
def rag_docs():
    return _guard(lambda: registry.ensure_loaded("rag").documents())


@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    raw = await file.read()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        raise HTTPException(400, "Only UTF-8 text / markdown files are supported")
    return _guard(lambda: registry.run("rag", lambda m: m.add_document(file.filename or "upload.txt", text), "index"))


# ------------------------------------------------------------------ 3. LLM
class ChatReq(Schema):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 256
    temperature: float = 0.7
    use_lora: bool = True


@app.post("/api/llm/chat")
def llm_chat(r: ChatReq):
    return _guard(lambda: registry.run("llm", lambda m: m.chat(r.messages[-12:], r.max_new_tokens, r.temperature, r.use_lora)))


@app.post("/api/llm/stream")
def llm_stream(r: ChatReq):
    m = _guard(lambda: registry.ensure_loaded("llm"))

    def gen():
        with m.lock:
            t0 = time.perf_counter()
            for piece in m.stream(r.messages[-12:], r.max_new_tokens, r.temperature, r.use_lora):
                if isinstance(piece, dict):
                    piece["rss_mb"] = round(rss_mb(), 1)
                    piece["params"] = m.params
                    m.history.append({"t": round(time.time(), 2), "ms": round((time.perf_counter() - t0) * 1000, 1),
                                      "op": "stream", "rss_mb": piece["rss_mb"]})
                    m.last_used = time.time()
                    yield json.dumps({"type": "done", **piece}) + "\n"
                else:
                    yield json.dumps({"type": "token", "text": piece}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ 4. Whisper
@app.post("/api/whisper/transcribe")
async def whisper(file: UploadFile = File(...), language: str = Form("auto"), reference: str = Form("")):
    data = await file.read()
    return _guard(lambda: registry.run("whisper", lambda m: m.transcribe(data, language, reference or None)))


# ------------------------------------------------------------------ 5. YOLO
@app.post("/api/yolo/detect")
async def yolo(file: UploadFile = File(...), conf: float = Form(0.25), iou: float = Form(0.45)):
    data = await file.read()
    return _guard(lambda: registry.run("yolo", lambda m: m.detect(data, conf, iou)))


# ------------------------------------------------------------------ 6. BERT
class TextReq(Schema):
    text: str
    explain: bool = True


class TextsReq(Schema):
    texts: List[str]


@app.post("/api/bert/classify")
def bert(r: TextReq):
    return _guard(lambda: registry.run("bert", lambda m: m.classify(r.text, r.explain)))


@app.post("/api/bert/batch")
def bert_batch(r: TextsReq):
    texts = [t for t in r.texts if t.strip()][:200]
    return _guard(lambda: registry.run("bert", lambda m: {"results": m.classify_batch(texts)}, "batch"))


# ------------------------------------------------------------------ 7. CLIP
@app.post("/api/clip/analyze")
async def clip(file: UploadFile = File(...), labels: str = Form("")):
    data = await file.read()
    lbls = [l.strip() for l in labels.split(",") if l.strip()] or None
    return _guard(lambda: registry.run("clip", lambda m: m.analyze(data, lbls)))


# ------------------------------------------------------------------ 8. LSTM-AE
class AnomReq(Schema):
    series: Optional[List[float]] = None
    n: int = 600
    anomalies: int = 4
    seed: Optional[int] = None
    sensitivity: float = 1.0


@app.post("/api/lstm_ae/detect")
def lstm(r: AnomReq):
    n = max(100, min(r.n, 5000))
    return _guard(lambda: registry.run("lstm_ae", lambda m: m.detect(r.series, n, r.anomalies, r.seed, r.sensitivity)))


# ------------------------------------------------------------------ 9. Two-tower
class SearchReq(Schema):
    query: str
    k: int = 5


@app.post("/api/two_tower/search")
def two_tower(r: SearchReq):
    return _guard(lambda: registry.run("two_tower", lambda m: m.search(r.query, r.k)))


# ------------------------------------------------------------------ 10. MIDI
class MidiReq(Schema):
    scale: str = "major"
    key: int = 0
    bars: int = 8
    temperature: float = 0.9
    top_k: int = 12
    bpm: int = 110
    seed: Optional[int] = None


@app.post("/api/midi/generate")
def midi(r: MidiReq):
    bars = max(1, min(r.bars, 32))
    return _guard(lambda: registry.run("midi", lambda m: m.generate(r.scale, r.key, bars, r.temperature, r.top_k, r.bpm, r.seed)))


# ------------------------------------------------------------------ samples + static frontend
SAMPLES = config.DATA_DIR / "samples"


@app.get("/api/samples")
def samples():
    out = []
    for p in sorted(SAMPLES.glob("*")):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".wav", ".mp3"):
            out.append({"name": p.name, "url": f"/api/samples/{p.name}",
                        "kind": "audio" if p.suffix.lower() in (".wav", ".mp3") else "image"})
    return out


@app.get("/api/samples/{name}")
def sample(name: str):
    p = (SAMPLES / name).resolve()
    if SAMPLES.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
