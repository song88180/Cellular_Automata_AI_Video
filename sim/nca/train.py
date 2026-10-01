#!/usr/bin/env python3
"""Dev-only: train a Growing Neural CA ("regenerating" variant) with PyTorch
and export weights for the pure-numpy runtime in sim/nca/model.py.

Follows the reference notebook of Mordvintsev et al., "Growing Neural
Cellular Automata", Distill 2020: 40x40 emoji target padded by 16, 16
channels, identity+Sobel perception, 48->128->16 update MLP with a
zero-initialized last layer, fire rate 0.5, sample pool of 1024 with the
worst sample reseeded and 3 samples circle-damaged per batch, 64-96 CA steps
per iteration, Adam lr 2e-3 -> 2e-4 at step 2000, per-parameter gradient
normalization.

Usage (from the project root, with the dev venv):
  .venv/bin/python -m sim.nca.train [--steps 8000] [--emoji 1f98e]
"""
import argparse
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sim.nca import model as np_model

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
TARGET_SIZE = 40
TARGET_PADDING = 16
CHANNELS = np_model.CHANNELS
HIDDEN = 128
FIRE_RATE = 0.5
POOL_SIZE = 1024
BATCH_SIZE = 8
DAMAGE_N = 3
EMOJI_URL = "https://raw.githubusercontent.com/googlefonts/noto-emoji/main/2D/png/128/emoji_u{code}.png"


def load_target(code: str) -> np.ndarray:
    """(H, W, 4) float32 RGBA in [0, 1], RGB premultiplied by alpha, padded."""
    path = WEIGHTS_DIR / f"target_emoji_u{code}.png"
    if not path.exists():
        with urllib.request.urlopen(EMOJI_URL.format(code=code)) as r:
            path.write_bytes(r.read())
    img = Image.open(io.BytesIO(path.read_bytes())).convert("RGBA")
    img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:]
    p = TARGET_PADDING
    return np.pad(a, ((p, p), (p, p), (0, 0)))


class TorchNCA(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Conv2d(CHANNELS * 3, HIDDEN, 1)
        self.w2 = nn.Conv2d(HIDDEN, CHANNELS, 1, bias=False)
        nn.init.zeros_(self.w2.weight)
        k = torch.stack([torch.from_numpy(np_model.SOBEL_X), torch.from_numpy(np_model.SOBEL_Y)])
        self.register_buffer("kernels", k[:, None].repeat(1, CHANNELS, 1, 1)[:, :, None])  # (2, C, 1, 3, 3)

    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kernels[0], padding=1, groups=CHANNELS)
        gy = F.conv2d(x, self.kernels[1], padding=1, groups=CHANNELS)
        return torch.cat([x, gx, gy], dim=1)

    @staticmethod
    def alive(x: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x[:, 3:4], 3, stride=1, padding=1) > np_model.ALIVE_THRESHOLD

    def forward(self, x: torch.Tensor, update_mask: torch.Tensor = None) -> torch.Tensor:
        pre = self.alive(x)
        dx = self.w2(F.relu(self.w1(self.perceive(x))))
        if update_mask is None:
            update_mask = torch.rand_like(x[:, :1]) <= FIRE_RATE
        x = x + dx * update_mask.float()
        return x * (pre & self.alive(x)).float()

    def export(self, path: Path) -> None:
        np.savez(
            path,
            W1=self.w1.weight[:, :, 0, 0].T.detach().cpu().numpy(),
            b1=self.w1.bias.detach().cpu().numpy(),
            W2=self.w2.weight[:, :, 0, 0].T.detach().cpu().numpy(),
            fire_rate=np.float32(FIRE_RATE),
        )


def circle_damage(n: int, h: int, w: int, device) -> torch.Tensor:
    """(n, 1, h, w) multiplicative masks with a random disc zeroed out."""
    ys = torch.linspace(-1, 1, h, device=device)[None, :, None]
    xs = torch.linspace(-1, 1, w, device=device)[None, None, :]
    center = torch.rand(2, n, 1, 1, device=device) - 0.5
    r = torch.rand(n, 1, 1, device=device) * 0.3 + 0.1
    inside = ((xs - center[0]) / r) ** 2 + ((ys - center[1]) / r) ** 2 < 1.0
    return (~inside).float()[:, None]


def per_sample_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


def train(steps: int, code: str, seed: int) -> None:
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_np = load_target(code)
    h, w = target_np.shape[:2]
    target = torch.from_numpy(target_np).permute(2, 0, 1)[None].to(device)
    seed_x = torch.from_numpy(np_model.seed_state(h, w)).permute(2, 0, 1)[None].to(device)

    nca = TorchNCA().to(device)
    opt = torch.optim.Adam(nca.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=0.1)
    pool = seed_x.repeat(POOL_SIZE, 1, 1, 1)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    for i in range(1, steps + 1):
        idx = torch.from_numpy(rng.choice(POOL_SIZE, BATCH_SIZE, replace=False)).to(device)
        x = pool[idx].clone()
        with torch.no_grad():
            order = per_sample_loss(x, target).argsort(descending=True)
        x, idx = x[order], idx[order]
        x[:1] = seed_x
        x[-DAMAGE_N:] *= circle_damage(DAMAGE_N, h, w, device)

        for _ in range(int(rng.integers(64, 97))):
            x = nca(x)
        loss = per_sample_loss(x, target).mean()

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in nca.parameters():
                p.grad /= p.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool[idx] = x.detach()

        if i % 100 == 0 or i == 1:
            print(f"step {i:5d}  loss {loss.item():.6f}  log10 {np.log10(loss.item()):.3f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        if i % 1000 == 0:
            nca.export(WEIGHTS_DIR / "model.npz")

    nca.export(WEIGHTS_DIR / "model.npz")
    print("exported", WEIGHTS_DIR / "model.npz")


def evaluate(target_np: np.ndarray, preview_path: Path = None) -> dict:
    """Grow + regenerate check using the *numpy* runtime on exported weights."""
    nca = np_model.NCA.load()
    rng = np.random.default_rng(0)
    h, w = target_np.shape[:2]

    def loss(x):
        return float(((x[..., :4] - target_np) ** 2).mean())

    x = np_model.seed_state(h, w)
    snaps = []
    for t in range(1, 301):
        x = nca.step(x, rng)
        if t in (20, 50, 100, 200, 300):
            snaps.append(x.copy())
    grown = loss(x)
    damaged = x.copy()
    damaged[:, w // 2:] = 0.0  # cut in half
    y = damaged.copy()
    for _ in range(200):
        y = nca.step(y, rng)
    snaps += [damaged, y]
    report = {"grown_loss": grown, "damaged_loss": loss(damaged), "regrown_loss": loss(y)}
    print(report)

    if preview_path is not None:
        bg = np.array([40, 42, 54], dtype=np.uint8)
        tiles = [np_model.to_rgb(s, bg) for s in snaps]
        tgt = (np.clip(target_np[..., :3] + (1 - target_np[..., 3:]) * bg / 255.0, 0, 1) * 255).astype(np.uint8)
        strip = np.concatenate([tgt] + tiles, axis=1)
        Image.fromarray(strip).resize((strip.shape[1] * 4, strip.shape[0] * 4), Image.NEAREST).save(preview_path)
        print("preview (target | t=20,50,100,200,300 | cut | regrown):", preview_path)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--emoji", default="1f98e", help="Noto emoji codepoint (default: lizard)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--preview", default="build/out/nca_preview.png")
    args = ap.parse_args()
    Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        evaluate(load_target(args.emoji), Path(args.preview))
    else:
        train(args.steps, args.emoji, args.seed)
        evaluate(load_target(args.emoji), Path(args.preview))
