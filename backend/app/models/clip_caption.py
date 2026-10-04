"""7. CLIP + small Transformer decoder for image captioning.

The decoder is trained *text-only* in CLIP's shared embedding space, following CapDec (Nukrai et al., 2022):
captions -> CLIP text encoder -> (noise injection) -> prefix -> Transformer decoder -> caption.
At inference the CLIP *image* embedding is fed in instead. This lets the captioner train in about
2 minutes on a laptop with no image dataset. scripts/train_clip_caption.py can retrain it on a larger
caption corpus.
"""
import io
import itertools
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .. import config
from ..core import BaseModel, torch_param_stats

CKPT = config.CKPT_DIR / "clip_caption_decoder.pt"
SAMPLES = config.DATA_DIR / "samples"

CONCEPTS = [
    "dog", "cat", "bird", "horse", "cow", "sheep", "person", "man", "woman", "child", "group of people",
    "car", "bus", "truck", "bicycle", "motorcycle", "train", "airplane", "boat", "traffic light",
    "laptop", "computer monitor", "keyboard", "smartphone", "book", "cup of coffee", "desk", "chair",
    "sofa", "bed", "table", "pizza", "sandwich", "salad", "cake", "fruit", "plate of food",
    "city street", "building", "skyscraper", "office", "kitchen", "living room", "park", "beach",
    "mountain", "forest", "river", "snow", "sunset", "night sky", "garden", "road", "field",
    "tennis racket", "football", "guitar", "piano", "flower", "tree", "umbrella", "clock",
]

# ----------------------------------------------------------------- synthetic caption corpus
_SUBJ = {
    "animal": ["a dog", "a brown dog", "a small puppy", "a cat", "a black cat", "a white cat", "a bird",
               "a horse", "two dogs", "a cow", "a group of sheep"],
    "person": ["a man", "a woman", "a young boy", "a little girl", "a group of people", "two people",
               "a person", "an office worker", "a chef", "a tennis player", "a musician"],
    "vehicle": ["a red car", "a white car", "a city bus", "a yellow taxi", "a bicycle", "a motorcycle",
                "a train", "an airplane", "a boat", "a truck"],
    "object": ["a laptop", "a computer monitor", "a cup of coffee", "a plate of food", "a pizza",
               "a bowl of salad", "a stack of books", "a smartphone", "a guitar", "a vase of flowers",
               "a birthday cake", "a wooden chair"],
}
_ACT = {
    "animal": ["sitting on", "running across", "lying on", "playing in", "standing in", "sleeping on", "walking through"],
    "person": ["standing in", "walking down", "sitting at", "working at", "smiling in", "playing in", "cooking in", "riding a bike on"],
    "vehicle": ["driving down", "parked on", "parked next to", "moving along", "stopped at"],
    "object": ["on", "sitting on", "placed on", "next to a window on"],
}
_PLACE = {
    "animal": ["the green grass", "a sunny park", "a sofa", "a wooden floor", "a sandy beach", "a snowy field",
               "a garden", "a bed", "a forest trail", "a muddy field"],
    "person": ["a busy city street", "a modern office", "a desk", "a kitchen", "a park", "a beach at sunset",
               "a tennis court", "a crowded room", "a quiet library", "a stage"],
    "vehicle": ["a busy city street", "a highway", "a parking lot", "the side of the road", "a traffic light",
                "a river", "a train station", "an airport runway", "a snowy road"],
    "object": ["a wooden table", "a white desk", "a kitchen counter", "an office desk", "a dining table",
               "a shelf", "a bed"],
}


def build_corpus(seed=0):
    random.seed(seed)
    caps = []
    for cat in _SUBJ:
        for s, a, p in itertools.product(_SUBJ[cat], _ACT[cat], _PLACE[cat]):
            caps.append(f"{s} {a} {p}")
    random.shuffle(caps)
    return caps


# ----------------------------------------------------------------- model
class PrefixCaptioner(nn.Module):
    def __init__(self, vocab, clip_dim=512, d=256, prefix_len=4, n_layer=3, n_head=4, max_len=24):
        super().__init__()
        self.cfg = dict(vocab=vocab, clip_dim=clip_dim, d=d, prefix_len=prefix_len, n_layer=n_layer,
                        n_head=n_head, max_len=max_len)
        self.vocab = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.prefix_len, self.max_len = prefix_len, max_len
        self.map = nn.Sequential(nn.Linear(clip_dim, d * prefix_len // 2), nn.Tanh(), nn.Linear(d * prefix_len // 2, d * prefix_len))
        self.tok = nn.Embedding(len(vocab), d)
        self.pos = nn.Embedding(prefix_len + max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(vocab))

    def forward(self, clip_emb, tokens):
        B, T = tokens.shape
        pre = self.map(clip_emb).view(B, self.prefix_len, -1)
        x = torch.cat([pre, self.tok(tokens)], 1)
        L = x.size(1)
        x = x + self.pos(torch.arange(L, device=x.device))
        mask = torch.triu(torch.full((L, L), float("-inf"), device=x.device), 1)
        h = self.ln(self.dec(x, mask=mask))
        return self.head(h[:, self.prefix_len - 1:-1])  # predict tokens[0..T-1]

    @torch.no_grad()
    def caption(self, clip_emb, max_len=20):
        bos, eos = self.stoi["<bos>"], self.stoi["<eos>"]
        toks = torch.tensor([[bos]], device=clip_emb.device)
        for _ in range(max_len):
            logits = self.forward(clip_emb, toks)[:, -1]
            logits[:, self.stoi["<pad>"]] = float("-inf")
            nxt = logits.argmax(-1, keepdim=True)
            if nxt.item() == eos:
                break
            toks = torch.cat([toks, nxt], 1)
        return " ".join(self.vocab[i] for i in toks[0, 1:].tolist())


def encode(vocab_stoi, cap, max_len):
    ids = [vocab_stoi["<bos>"]] + [vocab_stoi[w] for w in cap.split()] + [vocab_stoi["<eos>"]]
    ids = ids[:max_len + 1]
    return ids + [vocab_stoi["<pad>"]] * (max_len + 1 - len(ids))


def train_decoder(clip_model, processor, captions, steps=None, device="cpu", log=print):
    steps = steps or int(os.environ.get("CLIP_DECODER_STEPS", "1500"))
    vocab = ["<pad>", "<bos>", "<eos>"] + sorted({w for c in captions for w in c.split()})
    stoi = {w: i for i, w in enumerate(vocab)}
    with torch.no_grad():
        embs = []
        for i in range(0, len(captions), 256):
            t = processor(text=captions[i:i + 256], return_tensors="pt", padding=True).to(device)
            embs.append(F.normalize(clip_model.get_text_features(**t).float(), dim=-1).cpu())
        E = torch.cat(embs)
    max_len = 20
    T = torch.tensor([encode(stoi, c, max_len) for c in captions])
    model = PrefixCaptioner(vocab, clip_dim=E.size(1), max_len=max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    curve = []
    for step in range(1, steps + 1):
        ix = torch.randint(len(captions), (64,))
        e = E[ix].to(device)
        e = F.normalize(e + 0.1 * torch.randn_like(e), dim=-1)  # CapDec noise injection
        tok = T[ix].to(device)
        logits = model(e, tok[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, len(vocab)), tok[:, 1:].reshape(-1), ignore_index=stoi["<pad>"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0:
            curve.append((step, round(loss.item(), 4)))
            log(f"[clip-caption] step {step}/{steps} loss {loss.item():.4f}")
    model.eval()
    torch.save({"state": model.cpu().state_dict(), "cfg": model.cfg, "curve": curve, "n_captions": len(captions)}, CKPT)
    return model


class CLIPCaptioner(BaseModel):
    key = "clip"
    name = "CLIP + Transformer Decoder"
    family = "Vision-language (ViT + decoder)"
    task = "Image captioning and zero-shot tagging"
    modality = "vision"
    description = "CLIP ViT-B/32 embeds the image. A small 3-layer Transformer decoder (prefix-conditioned, trained CapDec-style in CLIP space) writes a caption, and CLIP also gives zero-shot tags and image-text similarity."
    heavy = True
    compute = "CPU/MPS · 151M CLIP + 3M decoder · ~600 MB"

    def _load(self):
        from transformers import CLIPModel, CLIPProcessor

        self.dev = config.DEVICE
        self.m_clip = CLIPModel.from_pretrained(config.CLIP_MODEL).to(self.dev).eval()
        self.m_proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        if CKPT.exists():
            ck = torch.load(CKPT, map_location="cpu")
            self.m_dec = PrefixCaptioner(**ck["cfg"])
            self.m_dec.load_state_dict(ck["state"])
            self.curve = ck.get("curve", [])
        else:
            self.notes = ["Caption decoder trained on first load (CapDec text-only)."]
            self.m_dec = train_decoder(self.m_clip, self.m_proc, build_corpus(), device=self.dev)
            self.curve = torch.load(CKPT, map_location="cpu").get("curve", [])
        self.m_dec.to(self.dev).eval()
        with torch.no_grad():
            t = self.m_proc(text=[f"a photo of a {c}" for c in CONCEPTS], return_tensors="pt", padding=True).to(self.dev)
            self.concept_emb = F.normalize(self.m_clip.get_text_features(**t).float(), dim=-1)
        self.variant = config.CLIP_MODEL.split("/")[-1] + " + 3L decoder"
        s = torch_param_stats(self.m_clip, self.m_dec)
        self.params, self.param_mb = s["params"], s["param_mb"]

    @torch.no_grad()
    def image_embedding(self, img: Image.Image):
        px = self.m_proc(images=img.convert("RGB"), return_tensors="pt").to(self.dev)
        return F.normalize(self.m_clip.get_image_features(**px).float(), dim=-1)

    @torch.no_grad()
    def text_embedding(self, texts):
        t = self.m_proc(text=texts, return_tensors="pt", padding=True).to(self.dev)
        return F.normalize(self.m_clip.get_text_features(**t).float(), dim=-1)

    def analyze_image(self, img: Image.Image, labels=None, top_k=5):
        t0 = time.perf_counter()
        e = self.image_embedding(img)
        enc_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        cap = self.m_dec.caption(e)
        dec_ms = (time.perf_counter() - t1) * 1000
        logits = 100 * e @ self.concept_emb.T
        probs = logits.softmax(-1)[0]
        top = probs.topk(top_k)
        tags = [{"label": CONCEPTS[i], "score": round(float(s), 4)} for s, i in zip(top.values, top.indices)]
        res = {"caption": cap, "tags": tags, "clip_score": round(float((e @ self.text_embedding([cap]).T).item()) * 100, 2),
               "encoder_ms": round(enc_ms, 1), "decoder_ms": round(dec_ms, 1)}
        if labels:
            le = self.text_embedding([f"a photo of {l}" for l in labels])
            p = (100 * e @ le.T).softmax(-1)[0]
            res["zero_shot"] = sorted([{"label": l, "score": round(float(s), 4)} for l, s in zip(labels, p)],
                                      key=lambda x: -x["score"])
        return res

    def analyze(self, data: bytes, labels=None):
        return self.analyze_image(Image.open(io.BytesIO(data)), labels)

    def benchmark(self):
        from ultralytics.utils import ASSETS

        cases = [(ASSETS / "bus.jpg", "bus"), (ASSETS / "zidane.jpg", "man")]
        for p in sorted(SAMPLES.glob("*.png")) + sorted(SAMPLES.glob("*.jpg")):
            stem = p.stem.split("_")[0]
            match = {"dog": "dog", "cat": "cat", "street": "city street", "office": "laptop", "food": "plate of food"}.get(stem)
            if match:
                cases.append((p, match))
        top1 = top5 = 0
        scores, lat = [], []
        for path, gold in cases:
            t0 = time.perf_counter()
            r = self.analyze_image(Image.open(path))
            lat.append((time.perf_counter() - t0) * 1000)
            labels = [t["label"] for t in r["tags"]]
            top1 += labels[0] == gold or (gold == "man" and labels[0] in ("person", "man"))
            top5 += gold in labels
            scores.append(r["clip_score"])
        n = len(cases)
        return {"primary": {"name": "Zero-shot top-5 acc", "value": round(top5 / n, 3), "better": "higher"},
                "metrics": {"top1": round(top1 / n, 3), "avg_caption_clipscore": round(sum(scores) / n, 2),
                            "avg_ms": round(sum(lat) / n, 1), "images": n}}
