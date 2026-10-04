"""10. Tiny Transformer for symbolic (MIDI) music generation."""
import base64
import io
import math
import os
import random
import struct

import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from ..core import BaseModel, torch_param_stats
from .tiny_gpt import TinyGPT, evaluate_lm, train_lm

CKPT = config.CKPT_DIR / "midi_transformer.pt"
LOW, HIGH = 48, 84  # C3..C6
DURS = [1, 2, 3, 4, 6, 8, 12, 16]  # in sixteenth notes
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# ---- vocabulary
VOCAB = ["<pad>", "<bos>", "<eos>", "<bar>"] + [f"<{s}>" for s in SCALES] + [f"<key{k}>" for k in range(12)] \
        + [f"P{p}" for p in range(LOW, HIGH + 1)] + ["REST"] + [f"D{d}" for d in DURS]
STOI = {t: i for i, t in enumerate(VOCAB)}
PITCH_IDS = [STOI[f"P{p}"] for p in range(LOW, HIGH + 1)] + [STOI["REST"]]
DUR_IDS = [STOI[f"D{d}"] for d in DURS]


def scale_pitches(root, scale):
    pcs = {(root + i) % 12 for i in SCALES[scale]}
    return [p for p in range(LOW, HIGH + 1) if p % 12 in pcs]


RHYTHMS = [[4, 4, 4, 4], [2, 2, 4, 4, 4], [4, 2, 2, 8], [6, 2, 4, 4], [3, 1, 4, 4, 4], [8, 4, 4],
           [2, 2, 2, 2, 4, 4], [4, 4, 8], [2, 2, 2, 2, 2, 2, 4], [6, 2, 6, 2], [16]]


def synth_melody(rng: random.Random):
    """Procedural training data: motif-based 8-bar phrases with stepwise motion, ending on the tonic."""
    scale = rng.choice(list(SCALES))
    root = rng.randrange(12)
    pitches = scale_pitches(root, scale)
    tonics = [i for i, p in enumerate(pitches) if p % 12 == root and 55 <= p <= 76] or [len(pitches) // 2]
    pos = rng.choice(tonics)

    def bar(pos):
        notes = []
        for d in rng.choice(RHYTHMS):
            if rng.random() < 0.06:
                notes.append(("REST", d)); continue
            step = rng.choices([-2, -1, 0, 1, 2, 3, -3, 4], [8, 25, 6, 25, 8, 4, 4, 2])[0]
            pos = min(max(pos + step, 0), len(pitches) - 1)
            notes.append((pitches[pos], d))
        return notes, pos

    motif_a, pos = bar(pos)
    motif_b, pos = bar(pos)
    seq = [motif_a, motif_b, motif_a]
    var, pos = bar(pos); seq.append(var)
    seq.append([(p if p == "REST" else min(max(p + rng.choice([0, 0, 2, -1]), LOW), HIGH), d) for p, d in motif_a])
    for _ in range(2):
        b, pos = bar(pos); seq.append(b)
    end = min(tonics, key=lambda t: abs(t - pos))
    seq.append([(pitches[min(end + 1, len(pitches) - 1)], 4), (pitches[end], 12)])
    toks = ["<bos>", f"<{scale}>", f"<key{root}>"]
    for b in seq:
        toks.append("<bar>")
        for p, d in b:
            if p != "REST" and p not in pitches:  # snap variations back into the scale
                p = min(pitches, key=lambda q: abs(q - p))
            toks += ["REST" if p == "REST" else f"P{p}", f"D{d}"]
    toks.append("<eos>")
    return toks


def build_data(n=3000, seed=0):
    rng = random.Random(seed)
    ids = []
    for _ in range(n):
        ids += [STOI[t] for t in synth_melody(rng)]
    return torch.tensor(ids, dtype=torch.long)


def train_and_save(steps=None, log=print):
    steps = steps or int(os.environ.get("MIDI_STEPS", "1500"))
    data = build_data(3000, 0)
    val = build_data(300, 99)
    model = TinyGPT(len(VOCAB), block_size=192, n_layer=4, n_head=4, n_embd=128, dropout=0.1)
    curve = train_lm(model, data, steps, 32, 2e-3, config.DEVICE, log=log, val_data=val)
    torch.save({"state": model.cpu().state_dict(), "cfg": model.cfg, "curve": curve}, CKPT)


# ---- MIDI writer (type 0)
def _vlq(n):
    out = [n & 0x7F]
    while n > 127:
        n >>= 7
        out.insert(0, (n & 0x7F) | 0x80)
    return bytes(out)


def notes_to_midi(notes, bpm=110, tpq=480, program=0):
    ev = []
    for n in notes:
        on = int(n["start"] * tpq); off = int((n["start"] + n["dur"]) * tpq)
        ev.append((on, 1, bytes([0x90, n["pitch"], n.get("vel", 90)])))
        ev.append((off, 0, bytes([0x80, n["pitch"], 0])))
    ev.sort(key=lambda e: (e[0], e[1]))
    trk = b"\x00\xff\x51\x03" + struct.pack(">I", int(60_000_000 / bpm))[1:]
    trk += b"\x00" + bytes([0xC0, program])
    t = 0
    for tick, _, msg in ev:
        trk += _vlq(tick - t) + msg; t = tick
    trk += b"\x00\xff\x2f\x00"
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq) + b"MTrk" + struct.pack(">I", len(trk)) + trk


class MidiTransformer(BaseModel):
    key = "midi"
    name = "Tiny Music Transformer"
    family = "Decoder-only Transformer"
    task = "Symbolic MIDI music generation"
    modality = "music"
    description = "A ~0.8M-parameter Transformer trained on event tokens (pitch, duration, bar, key, scale). Grammar-constrained sampling produces melodies you can play in the browser or download as .mid files."
    compute = "CPU/MPS · 0.8M params · trains in ~1 min"

    def _load(self):
        if not CKPT.exists():
            self.notes = ["Trained on procedurally generated melodies on first load."]
            train_and_save()
        ck = torch.load(CKPT, map_location="cpu")
        self.m_model = TinyGPT(**ck["cfg"])
        self.m_model.load_state_dict(ck["state"])
        self.m_model.to(config.DEVICE).eval()
        self.curve = ck["curve"]
        self.variant = "4L-128d event transformer"
        s = torch_param_stats(self.m_model)
        self.params, self.param_mb = s["params"], s["param_mb"]

    @torch.no_grad()
    def _sample(self, scale, key, bars, temperature, top_k, seed):
        if seed is not None:
            torch.manual_seed(int(seed))
        ids = [STOI["<bos>"], STOI[f"<{scale}>"], STOI[f"<key{key}>"], STOI["<bar>"]]
        x = torch.tensor([ids], device=config.DEVICE)
        n_bars, expect_dur = 1, False
        allowed_pitch = torch.full((len(VOCAB),), float("-inf"), device=config.DEVICE)
        allowed_pitch[PITCH_IDS] = 0
        allowed_pitch[STOI["<bar>"]] = 0
        allowed_pitch[STOI["<eos>"]] = 0
        allowed_dur = torch.full((len(VOCAB),), float("-inf"), device=config.DEVICE)
        allowed_dur[DUR_IDS] = 0
        for _ in range(bars * 24):
            logits, _ = self.m_model(x[:, -self.m_model.block_size:])
            lg = logits[0, -1] / max(temperature, 1e-3) + (allowed_dur if expect_dur else allowed_pitch)
            if not expect_dur and n_bars < bars:
                lg[STOI["<eos>"]] = float("-inf")
            if top_k:
                v, _ = torch.topk(lg, min(top_k, len(VOCAB)))
                lg[lg < v[-1]] = float("-inf")
            nxt = int(torch.multinomial(F.softmax(lg, -1), 1))
            if nxt == STOI["<eos>"]:
                break
            if nxt == STOI["<bar>"]:
                n_bars += 1
                if n_bars > bars:
                    break
            else:
                expect_dur = not expect_dur
            x = torch.cat([x, torch.tensor([[nxt]], device=config.DEVICE)], 1)
        return [VOCAB[i] for i in x[0].tolist()]

    def generate(self, scale="major", key=0, bars=8, temperature=0.9, top_k=12, bpm=110, seed=None):
        scale = scale if scale in SCALES else "major"
        toks = self._sample(scale, int(key) % 12, int(bars), float(temperature), int(top_k), seed)
        notes, t, pending = [], 0.0, None
        for tok in toks:
            if tok.startswith("P") or tok == "REST":
                pending = tok
            elif tok.startswith("D") and pending:
                d = int(tok[1:]) / 4  # quarter-note beats
                if pending != "REST":
                    notes.append({"pitch": int(pending[1:]), "start": t, "dur": d, "vel": 80 + random.randint(-10, 15)})
                t += d; pending = None
        midi = notes_to_midi(notes, bpm=bpm)
        return {"notes": notes, "tokens": toks, "bpm": bpm, "total_beats": t,
                "key_name": f"{NOTE_NAMES[int(key) % 12]} {scale}", "midi_base64": base64.b64encode(midi).decode(),
                "analysis": self._analyze(notes, int(key) % 12, scale), "training_curve": self.curve}

    @staticmethod
    def _analyze(notes, key, scale):
        if not notes:
            return {}
        pcs = {(key + i) % 12 for i in SCALES[scale]}
        in_scale = sum(n["pitch"] % 12 in pcs for n in notes) / len(notes)
        hist = np.bincount([n["pitch"] % 12 for n in notes], minlength=12) / len(notes)
        ent = float(-(hist[hist > 0] * np.log2(hist[hist > 0])).sum())
        ps = [n["pitch"] for n in notes]
        tri = [tuple(ps[i:i + 3]) for i in range(len(ps) - 2)]
        return {"in_scale_ratio": round(in_scale, 3), "pitch_class_entropy": round(ent, 3),
                "unique_trigram_ratio": round(len(set(tri)) / max(1, len(tri)), 3),
                "pitch_range": max(ps) - min(ps), "n_notes": len(notes)}

    def benchmark(self):
        val = build_data(300, 99)
        loss = evaluate_lm(self.m_model, val, 32, config.DEVICE, iters=10)
        stats = [self.generate(s, k, 8, seed=k * 7 + i)["analysis"] for i, (s, k) in
                 enumerate([("major", 0), ("minor", 9), ("pentatonic", 7), ("blues", 2), ("dorian", 4), ("major", 5)])]
        avg = lambda k: round(float(np.mean([s[k] for s in stats])), 3)  # noqa: E731
        return {"primary": {"name": "In-scale note ratio", "value": avg("in_scale_ratio"), "better": "higher"},
                "metrics": {"val_perplexity": round(math.exp(loss), 3), "pitch_class_entropy": avg("pitch_class_entropy"),
                            "unique_trigram_ratio": avg("unique_trigram_ratio"), "samples": len(stats)}}
