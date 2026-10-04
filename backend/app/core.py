"""Base class + registry that handles lazy loading, eviction and metric collection."""
from __future__ import annotations

import gc
import threading
import time
import traceback
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import psutil

from . import config

_PROC = psutil.Process()


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6


def torch_param_stats(*modules) -> Dict[str, float]:
    """Count parameters / bytes over any number of torch modules."""
    n, b = 0, 0
    for m in modules:
        if m is None or not hasattr(m, "parameters"):
            continue
        for p in m.parameters():
            n += p.numel()
            b += p.numel() * p.element_size()
    return {"params": n, "param_mb": b / 1e6}


def free_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class BaseModel:
    key: str = ""
    name: str = ""
    family: str = ""  # e.g. "Transformer", "CNN", "RNN"
    task: str = ""
    modality: str = ""  # text / audio / vision / time-series / music
    description: str = ""
    heavy: bool = False  # large pretrained weights -> subject to eviction
    compute: str = ""  # human-readable compute requirement

    def __init__(self) -> None:
        self.status = "unloaded"
        self.variant: Optional[str] = None  # which weights are actually serving
        self.error: Optional[str] = None
        self.params = 0
        self.param_mb = 0.0
        self.load_seconds: Optional[float] = None
        self.load_rss_delta_mb: Optional[float] = None
        self.last_used = 0.0
        self.history: deque = deque(maxlen=50)
        self.benchmark_result: Optional[Dict[str, Any]] = None
        self.lock = threading.RLock()
        self.notes: List[str] = []

    # ---- to implement -------------------------------------------------
    def _load(self) -> None:
        raise NotImplementedError

    def _unload(self) -> None:
        for attr in list(vars(self)):
            if attr.startswith("m_"):
                setattr(self, attr, None)

    def benchmark(self) -> Dict[str, Any]:
        return {}

    # ---- public ---------------------------------------------------------
    def info(self) -> Dict[str, Any]:
        h = list(self.history)
        lat = [x["ms"] for x in h]
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "task": self.task,
            "modality": self.modality,
            "description": self.description,
            "heavy": self.heavy,
            "compute": self.compute,
            "status": self.status,
            "variant": self.variant,
            "error": self.error,
            "device": config.DEVICE,
            "params": self.params,
            "param_mb": round(self.param_mb, 1),
            "load_seconds": self.load_seconds,
            "load_rss_delta_mb": self.load_rss_delta_mb,
            "calls": len(h),
            "avg_ms": round(sum(lat) / len(lat), 1) if lat else None,
            "last_ms": lat[-1] if lat else None,
            "history": h[-20:],
            "benchmark": self.benchmark_result,
            "notes": self.notes,
        }


class Registry:
    def __init__(self) -> None:
        self.models: Dict[str, BaseModel] = {}
        self._lock = threading.Lock()

    def register(self, m: BaseModel) -> None:
        self.models[m.key] = m

    def get(self, key: str) -> BaseModel:
        if key not in self.models:
            raise KeyError(key)
        return self.models[key]

    def _evict_if_needed(self, incoming: BaseModel) -> None:
        if not incoming.heavy:
            return
        resident = [m for m in self.models.values() if m.heavy and m.status == "ready" and m is not incoming]
        resident.sort(key=lambda m: m.last_used)
        while len(resident) >= config.MAX_RESIDENT_HEAVY:
            victim = resident.pop(0)
            self.unload(victim.key)

    def ensure_loaded(self, key: str) -> BaseModel:
        m = self.get(key)
        with m.lock:
            if m.status == "ready":
                return m
            with self._lock:
                self._evict_if_needed(m)
            m.status, m.error = "loading", None
            before, t0 = rss_mb(), time.perf_counter()
            try:
                m._load()
                m.status = "ready"
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                m.status, m.error = "error", f"{type(e).__name__}: {e}"
                raise
            finally:
                m.load_seconds = round(time.perf_counter() - t0, 2)
                m.load_rss_delta_mb = round(rss_mb() - before, 1)
            m.last_used = time.time()
            return m

    def unload(self, key: str) -> None:
        m = self.get(key)
        with m.lock:
            m._unload()
            m.status, m.variant = "unloaded", None
            free_accelerator_memory()

    def run(self, key: str, fn: Callable[[BaseModel], Dict[str, Any]], label: str = "infer") -> Dict[str, Any]:
        m = self.ensure_loaded(key)
        with m.lock:
            before = rss_mb()
            t0 = time.perf_counter()
            out = fn(m)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            m.last_used = time.time()
            rec = {"t": round(time.time(), 2), "ms": ms, "op": label, "rss_mb": round(rss_mb(), 1)}
            m.history.append(rec)
        out = dict(out)
        out["metrics"] = {
            "inference_ms": ms,
            "rss_mb": rec["rss_mb"],
            "rss_delta_mb": round(rec["rss_mb"] - before, 1),
            "variant": m.variant,
            "device": config.DEVICE,
            "params": m.params,
        }
        return out


registry = Registry()
