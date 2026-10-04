"""4. Whisper speech recognition (base, with tiny as fallback)."""
import re
import shutil
import subprocess
import time

import numpy as np

from .. import config
from ..core import BaseModel, torch_param_stats

SAMPLE_TEXT = ("Welcome to Nimbus Corp. Please submit your expense claims within thirty days, "
               "and remember to enable multi factor authentication on your laptop.")
SAMPLE_WAV = config.DATA_DIR / "samples" / "speech_sample.wav"


def decode_audio(data: bytes, sr: int = 16000) -> np.ndarray:
    """Decode any container/codec to mono float32 PCM via ffmpeg."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio decoding (brew install ffmpeg)")
    p = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, capture_output=True, check=True,
    )
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def _norm(t):
    return re.sub(r"[^a-z0-9' ]+", " ", t.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1)
    d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[-1, -1] / max(1, len(r)))


def ensure_sample():
    if SAMPLE_WAV.exists():
        return True
    if shutil.which("say"):
        SAMPLE_WAV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["say", "-o", str(SAMPLE_WAV), "--data-format=LEI16@16000", SAMPLE_TEXT], check=False)
    return SAMPLE_WAV.exists()


class WhisperASR(BaseModel):
    key = "whisper"
    name = "Whisper"
    family = "Encoder-decoder Transformer"
    task = "Speech recognition"
    modality = "audio"
    description = "OpenAI Whisper turns log-Mel spectrograms into text. Records from your microphone or takes audio uploads, with timestamps and word error rate scoring."
    heavy = True
    compute = "CPU/MPS · base 74M params · ~290 MB"

    def _load(self):
        from transformers import pipeline

        device = "cpu" if config.DEVICE == "mps" else config.DEVICE  # fp32 CPU is most stable for whisper on Mac
        last = None
        for mid in (config.WHISPER_MODEL, config.WHISPER_FALLBACK):
            try:
                self.m_pipe = pipeline("automatic-speech-recognition", model=mid, device=device)
                self.variant = mid.split("/")[-1]
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"Whisper unavailable: {last}")
        s = torch_param_stats(self.m_pipe.model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    def transcribe(self, audio_bytes: bytes, language: str = None, reference: str = None):
        t0 = time.perf_counter()
        audio = decode_audio(audio_bytes)
        decode_ms = (time.perf_counter() - t0) * 1000
        duration = len(audio) / 16000
        gk = {"task": "transcribe"}
        if language and language != "auto":
            gk["language"] = language
        t1 = time.perf_counter()
        out = self.m_pipe({"raw": audio, "sampling_rate": 16000}, return_timestamps=True,
                          chunk_length_s=30, generate_kwargs=gk)
        asr_ms = (time.perf_counter() - t1) * 1000
        chunks = [{"text": c["text"].strip(), "start": c["timestamp"][0], "end": c["timestamp"][1]}
                  for c in out.get("chunks", [])]
        # Waveform envelope for the UI
        env = []
        if len(audio):
            n = 120
            step = max(1, len(audio) // n)
            env = [round(float(np.abs(audio[i:i + step]).max()), 3) for i in range(0, step * n, step)][:n]
        res = {"text": out["text"].strip(), "chunks": chunks, "duration_s": round(duration, 2),
               "decode_ms": round(decode_ms, 1), "asr_ms": round(asr_ms, 1),
               "real_time_factor": round((asr_ms / 1000) / max(duration, 1e-3), 3), "waveform": env}
        if reference:
            res["wer"] = round(wer(reference, res["text"]), 4)
        return res

    def benchmark(self):
        if not ensure_sample():
            return {"primary": {"name": "WER", "value": None, "better": "lower"},
                    "metrics": {"note": "No sample audio found (put a wav at data/samples/speech_sample.wav)"}}
        r = self.transcribe(SAMPLE_WAV.read_bytes(), "en", SAMPLE_TEXT)
        return {"primary": {"name": "Word error rate", "value": r["wer"], "better": "lower"},
                "metrics": {"real_time_factor": r["real_time_factor"], "asr_ms": r["asr_ms"],
                            "audio_s": r["duration_s"], "hypothesis": r["text"]}}
