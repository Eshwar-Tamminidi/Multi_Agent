"""5. YOLOv8 real-time object detection."""
import io
import time

from PIL import Image

from .. import config
from ..core import BaseModel, torch_param_stats

# Expected objects in the sample images shipped with ultralytics (label-level ground truth)
EXPECTED = {"bus.jpg": {"person": 4, "bus": 1}, "zidane.jpg": {"person": 2, "tie": 1}}


class YOLODetector(BaseModel):
    key = "yolo"
    name = "YOLOv8"
    family = "Single-stage CNN detector"
    task = "Object detection"
    modality = "vision"
    description = "Ultralytics YOLOv8-nano finds 80 COCO object classes in a single forward pass and returns boxes, labels and confidences in real time."
    compute = "CPU/MPS · 3.2M params · ~6 MB"

    def _load(self):
        from ultralytics import YOLO

        path = config.CKPT_DIR / config.YOLO_WEIGHTS
        self.m_model = YOLO(str(path) if path.exists() else config.YOLO_WEIGHTS)
        if not path.exists():
            # ultralytics downloads to CWD; move into checkpoints for tidiness
            import shutil
            from pathlib import Path

            if Path(config.YOLO_WEIGHTS).exists():
                shutil.move(config.YOLO_WEIGHTS, path)
        self.variant = config.YOLO_WEIGHTS.replace(".pt", "")
        s = torch_param_stats(self.m_model.model)
        self.params, self.param_mb = s["params"], s["param_mb"]
        self.device = "mps" if config.DEVICE == "mps" else ("0" if config.DEVICE == "cuda" else "cpu")

    def detect_image(self, img: Image.Image, conf=0.25, iou=0.45):
        img = img.convert("RGB")
        r = self.m_model.predict(img, conf=float(conf), iou=float(iou), device=self.device, verbose=False)[0]
        W, H = img.size
        dets, counts = [], {}
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            label = r.names[int(b.cls)]
            counts[label] = counts.get(label, 0) + 1
            dets.append({"label": label, "confidence": round(float(b.conf), 3),
                         "box": [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]})
        sp = r.speed  # ms for preprocess / inference / postprocess
        return {"detections": dets, "counts": counts, "width": W, "height": H,
                "speed_ms": {k: round(v, 2) for k, v in sp.items()}}

    def detect(self, data: bytes, conf=0.25, iou=0.45):
        return self.detect_image(Image.open(io.BytesIO(data)), conf, iou)

    def benchmark(self):
        from ultralytics.utils import ASSETS

        tp = fp = fn = 0
        times = []
        for name, exp in EXPECTED.items():
            img = Image.open(ASSETS / name)
            self.detect_image(img)  # warm-up
            t0 = time.perf_counter()
            r = self.detect_image(img)
            times.append((time.perf_counter() - t0) * 1000)
            got = r["counts"]
            for lbl in set(exp) | set(got):
                e, g = exp.get(lbl, 0), got.get(lbl, 0)
                tp += min(e, g)
                fp += max(0, g - e)
                fn += max(0, e - g)
        p = tp / max(1, tp + fp)
        rc = tp / max(1, tp + fn)
        f1 = 2 * p * rc / max(1e-9, p + rc)
        ms = sum(times) / len(times)
        return {"primary": {"name": "Count-level F1", "value": round(f1, 3), "better": "higher"},
                "metrics": {"precision": round(p, 3), "recall": round(rc, 3), "avg_ms": round(ms, 1),
                            "fps": round(1000 / ms, 1), "images": len(EXPECTED)}}
