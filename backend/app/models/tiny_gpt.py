"""A minimal decoder-only Transformer (nanoGPT style) shared by Char-GPT and the MIDI generator."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.cfg = dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                        n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, banned=None):
        for _ in range(max_new_tokens):
            cond = idx[:, -self.block_size:]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if banned:
                logits[:, banned] = float("-inf")
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def train_lm(model, data: torch.Tensor, steps: int, batch_size: int, lr: float, device: str, log=None,
             val_data: torch.Tensor = None):
    """Simple causal-LM training loop on a 1-D token tensor. Returns list of (step, train_loss, val_loss)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    bs = model.block_size
    curve = []
    for step in range(1, steps + 1):
        ix = torch.randint(len(data) - bs - 1, (batch_size,))
        x = torch.stack([data[i:i + bs] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + bs + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            vl = evaluate_lm(model, val_data, batch_size, device) if val_data is not None else None
            curve.append((step, round(loss.item(), 4), vl))
            if log:
                log(f"step {step}/{steps} loss {loss.item():.4f} val {vl}")
            model.train()
    model.eval()
    return curve


@torch.no_grad()
def evaluate_lm(model, data, batch_size, device, iters=10):
    model.eval()
    bs = model.block_size
    if data is None or len(data) <= bs + 1:
        return None
    losses = []
    g = torch.Generator().manual_seed(0)
    for _ in range(iters):
        ix = torch.randint(len(data) - bs - 1, (batch_size,), generator=g)
        x = torch.stack([data[i:i + bs] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + bs + 1] for i in ix]).to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    return round(sum(losses) / len(losses), 4)
