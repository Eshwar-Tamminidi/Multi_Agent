"""Benchmark all 10 models sequentially and write benchmarks.json."""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import registry, rss_mb
from app import config
import app.main # registers all models

def main():
    print(f"Starting comprehensive benchmark suite across all 10 models on device: {config.DEVICE}")
    bench_results = {}
    for key, model in registry.models.items():
        print(f"\n[{key}] Loading & benchmarking {model.name}...")
        try:
            m = registry.ensure_loaded(key)
            t0 = time.perf_counter()
            res = m.benchmark()
            duration = round(time.perf_counter() - t0, 2)
            res.update({
                "benchmark_seconds": duration,
                "timestamp": time.time(),
                "params": m.params,
                "param_mb": round(m.param_mb, 1),
                "load_seconds": m.load_seconds,
                "load_rss_delta_mb": m.load_rss_delta_mb,
                "variant": m.variant,
                "device": config.DEVICE,
                "avg_inference_ms": m.info().get("avg_ms"),
                "rss_mb": round(rss_mb(), 1)
            })
            model.benchmark_result = res
            bench_results[key] = res
            print(f"[{key}] Done in {duration}s -> {res.get('primary')}")
        except Exception as e:
            print(f"[{key}] Benchmark error: {e}")
            bench_results[key] = {"error": str(e)}

    out_file = config.CKPT_DIR / "benchmarks.json"
    out_file.write_text(json.dumps(bench_results, indent=2, default=str))
    print(f"\nAll benchmark results saved to {out_file}")

if __name__ == "__main__":
    main()
