import argparse
import copy
import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HOME = os.path.expanduser("~/ml_llm_pi/MouseSD-1")
CKPT = os.path.join(HOME, "mousesd1.pt")
SAMPLES = os.path.join(HOME, "samples")
OUT = os.path.join(HOME, "out")
DATA = os.path.join(HOME, "data")
MAXLEN = 64
T = 1000

if torch.cuda.is_available():
    DEV = "cuda"
elif torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cpu"


def alphas_cumprod():
    s = 0.008
    x = torch.linspace(0, T, T + 1)
    f = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    a = f / f[0]
    betas = (1 - a[1:] / a[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - betas, 0)


AB = alphas_cumprod().to(DEV)


def encode(text, stoi):
    ids = [1] + [stoi[c] for c in text.lower()[: MAXLEN - 1] if c in stoi]
    return ids + [0] * (MAXLEN - len(ids))


class Sin(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)


class Block(nn.Module):
    def __init__(self, i, o, e):
        super().__init__()
        self.n1 = nn.GroupNorm(8, i)
        self.c1 = nn.Conv2d(i, o, 3, padding=1)
        self.emb = nn.Linear(e, o * 2)
        self.n2 = nn.GroupNorm(8, o)
        self.c2 = nn.Conv2d(o, o, 3, padding=1)
        self.sk = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.emb(c)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.c2(F.silu(h))
        return h + self.sk(x)


class Attn(nn.Module):
    def __init__(self, c, e):
        super().__init__()
        self.n1 = nn.LayerNorm(c)
        self.n2 = nn.LayerNorm(c)
        self.sa = nn.MultiheadAttention(c, 4, batch_first=True)
        self.ca = nn.MultiheadAttention(c, 4, kdim=e, vdim=e, batch_first=True)

    def forward(self, x, txt, mask):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        q = self.n1(t)
        t = t + self.sa(q, q, q, need_weights=False)[0]
        q = self.n2(t)
        t = t + self.ca(q, txt, txt, key_padding_mask=mask, need_weights=False)[0]
        return t.transpose(1, 2).reshape(B, C, H, W)


class Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Conv2d(i, o, 3, padding=1)

    def forward(self, x):
        return self.c(F.interpolate(x, scale_factor=2, mode="nearest"))


class MouseSD(nn.Module):
    def __init__(self, vocab, ch=96, emb=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, MAXLEN, emb) * 0.02)
        layer = nn.TransformerEncoderLayer(emb, 4, emb * 4, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
        self.t = nn.Sequential(Sin(emb), nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inc = nn.Conv2d(3, ch, 3, padding=1)
        self.b1 = Block(ch, ch, emb)
        self.d1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.b2 = Block(ch, ch * 2, emb)
        self.d2 = nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1)
        self.b3 = Block(ch * 2, ch * 4, emb)
        self.a3 = Attn(ch * 4, emb)
        self.m1 = Block(ch * 4, ch * 4, emb)
        self.ma = Attn(ch * 4, emb)
        self.m2 = Block(ch * 4, ch * 4, emb)
        self.u3 = Block(ch * 8, ch * 4, emb)
        self.ua = Attn(ch * 4, emb)
        self.up2 = Up(ch * 4, ch * 2)
        self.u2 = Block(ch * 4, ch * 2, emb)
        self.up1 = Up(ch * 2, ch)
        self.u1 = Block(ch * 2, ch, emb)
        self.no = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x, t, tok, drop):
        null = torch.zeros_like(tok)
        null[:, 0] = 1
        tok = torch.where(drop[:, None], null, tok)
        mask = tok == 0
        txt = self.enc(self.tok(tok) + self.pos, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        c = self.t(t) + (txt * keep).sum(1) / keep.sum(1)
        h0 = self.inc(x)
        h1 = self.b1(h0, c)
        h2 = self.b2(self.d1(h1), c)
        h3 = self.a3(self.b3(self.d2(h2), c), txt, mask)
        m = self.m2(self.ma(self.m1(h3, c), txt, mask), c)
        u = self.ua(self.u3(torch.cat([m, h3], 1), c), txt, mask)
        u = self.u2(torch.cat([self.up2(u), h2], 1), c)
        u = self.u1(torch.cat([self.up1(u), h1], 1), c)
        return self.out(F.silu(self.no(u)))


@torch.no_grad()
def sample(model, tok, res, steps, guidance, seed):
    B = tok.shape[0]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 3, res, res, generator=g).to(DEV)
    ts = torch.linspace(T - 1, 0, steps).long().tolist()
    drop = torch.cat([torch.zeros(B), torch.ones(B)]).bool().to(DEV)
    tok2 = torch.cat([tok, tok])
    for i, t in enumerate(ts):
        a = AB[t]
        ap = AB[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=DEV)
        tt = torch.full((2 * B,), t, device=DEV)
        ec, eu = model(torch.cat([x, x]), tt, tok2, drop).chunk(2)
        e = eu + guidance * (ec - eu)
        x0 = ((x - (1 - a).sqrt() * e) / a.sqrt()).clamp(-1, 1)
        e = (x - a.sqrt() * x0) / (1 - a).sqrt()
        x = ap.sqrt() * x0 + (1 - ap).sqrt() * e
    return ((x.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()


def load_data(root, res):
    paths = []
    for ext in ("png", "jpg", "jpeg", "webp"):
        paths += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    paths.sort()
    imgs, caps = [], []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        w, h = im.size
        m = min(w, h)
        im = im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
        im = im.resize((res, res), Image.LANCZOS)
        imgs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
        txt = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(txt):
            cap = open(txt, encoding="utf-8").read().strip()
        else:
            parent = os.path.basename(os.path.dirname(p))
            cap = parent if os.path.abspath(os.path.dirname(p)) != os.path.abspath(root) else "image"
        caps.append(cap.lower())
    return imgs, caps


def save_grid(arr, path):
    n, h, w, _ = arr.shape
    canvas = Image.new("RGB", (w * n, h))
    for i in range(n):
        canvas.paste(Image.fromarray(arr[i]), (i * w, 0))
    canvas.save(path)


def train(a):
    os.makedirs(SAMPLES, exist_ok=True)
    ck = torch.load(CKPT, map_location=DEV) if os.path.exists(CKPT) else None
    res = ck["res"] if ck else a.res
    ch = ck["ch"] if ck else a.ch
    imgs, caps = load_data(a.data, res)
    if not imgs:
        print(f"no images in {a.data}")
        return
    chars = ck["chars"] if ck else sorted(set("".join(caps)))
    stoi = {c: i + 2 for i, c in enumerate(chars)}
    toks = torch.tensor([encode(c, stoi) for c in caps]).to(DEV)
    data = torch.stack(imgs).to(DEV)
    N = data.shape[0]
    model = MouseSD(len(chars) + 2, ch).to(DEV)
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    step = 0
    if ck:
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
    print(f"MouseSD-1 | {DEV} | {N} images | {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | step {step}")
    preview = list(dict.fromkeys(caps))[:4]
    ptok = torch.tensor([encode(c, stoi) for c in preview]).to(DEV)

    def save():
        torch.save(
            {"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
             "step": step, "chars": chars, "res": res, "ch": ch},
            CKPT,
        )

    t0 = time.time()
    run = 0.0
    try:
        while step < a.steps:
            idx = torch.randint(0, N, (a.batch,), device=DEV)
            x = data[idx].float() / 127.5 - 1
            flip = torch.rand(a.batch, device=DEV) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
            t = torch.randint(0, T, (a.batch,), device=DEV)
            n = torch.randn_like(x)
            ab = AB[t][:, None, None, None]
            xt = ab.sqrt() * x + (1 - ab).sqrt() * n
            drop = torch.rand(a.batch, device=DEV) < 0.1
            loss = F.mse_loss(model(xt, t, toks[idx], drop), n)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            d = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
            run = loss.item() if step == 1 else run * 0.98 + loss.item() * 0.02
            if step % 100 == 0:
                print(f"step {step} loss {run:.4f} {time.time() - t0:.0f}s")
            if step % 500 == 0:
                save()
            if step % 1000 == 0:
                save_grid(sample(ema, ptok, res, 30, 3.0, 0), os.path.join(SAMPLES, f"step_{step}.png"))
    except KeyboardInterrupt:
        pass
    save()
    print("saved", CKPT)


def gen(a):
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(CKPT, map_location=DEV)
    stoi = {c: i + 2 for i, c in enumerate(ck["chars"])}
    model = MouseSD(len(ck["chars"]) + 2, ck["ch"]).to(DEV)
    model.load_state_dict(ck["ema"])
    model.eval()
    seed = a.seed if a.seed >= 0 else random.randint(0, 2**31 - 1)
    tok = torch.tensor([encode(" ".join(a.prompt), stoi)] * a.n).to(DEV)
    arr = sample(model, tok, ck["res"], a.steps, a.guidance, seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for i in range(a.n):
        im = Image.fromarray(arr[i])
        if a.scale > 1:
            im = im.resize((ck["res"] * a.scale,) * 2, Image.BICUBIC)
        p = os.path.join(OUT, f"{stamp}_{seed}_{i}.png")
        im.save(p)
        print(p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default=DATA)
    t.add_argument("--res", type=int, default=64)
    t.add_argument("--ch", type=int, default=96)
    t.add_argument("--steps", type=int, default=50000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    g = sub.add_parser("gen")
    g.add_argument("prompt", nargs="+")
    g.add_argument("--n", type=int, default=4)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--guidance", type=float, default=3.0)
    g.add_argument("--seed", type=int, default=-1)
    g.add_argument("--scale", type=int, default=4)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else gen(a)


if __name__ == "__main__":
    main()
