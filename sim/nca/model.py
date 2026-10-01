"""Growing Neural Cellular Automata -- pure-numpy inference (no torch).

Re-implementation of Mordvintsev et al., "Growing Neural Cellular Automata",
Distill 2020 (https://distill.pub/2020/growing-ca/). Must stay numerically
identical to sim/nca/train.py (checked by tests/test_nca_model.py).

State: (H, W, 16) float32. Channels 0-2 = RGB premultiplied by alpha,
channel 3 = alpha ("alive" if > 0.1), channels 4-15 = hidden.
One step:
  1. perception = concat([x, sobel_x(x), sobel_y(x)])  -> 48 per cell (zero padding)
  2. dx = relu(perception @ W1 + b1) @ W2               -> 16 per cell
  3. x += dx * (random mask, each cell fires with p = fire_rate)
  4. x *= alive_before & alive_after   (alive = 3x3 max-pool of alpha > 0.1)
"""
from pathlib import Path
from typing import Optional

import numpy as np

CHANNELS = 16
ALIVE_THRESHOLD = 0.1
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "model.npz"

# 3x3 kernels, cross-correlation convention (same as torch conv2d / TF conv).
SOBEL_X = np.outer([1, 2, 1], [-1, 0, 1]).astype(np.float32) / 8.0
SOBEL_Y = SOBEL_X.T.copy()


def _correlate3x3(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Depthwise 3x3 cross-correlation of (H, W, C) with zero padding."""
    H, W = x.shape[:2]
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    out = np.zeros_like(x)
    for a in range(3):
        for b in range(3):
            if k[a, b] != 0:
                out += k[a, b] * p[a: a + H, b: b + W]
    return out


def alive_mask(x: np.ndarray) -> np.ndarray:
    a = np.pad(x[..., 3], 1)  # zero padding is safe: 0 < threshold
    H, W = x.shape[:2]
    m = np.max([a[i: i + H, j: j + W] for i in range(3) for j in range(3)], axis=0)
    return m > ALIVE_THRESHOLD


def perceive(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, _correlate3x3(x, SOBEL_X), _correlate3x3(x, SOBEL_Y)], axis=-1)


class NCA:
    def __init__(self, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, fire_rate: float = 0.5):
        self.W1 = W1.astype(np.float32)  # (48, 128)
        self.b1 = b1.astype(np.float32)  # (128,)
        self.W2 = W2.astype(np.float32)  # (128, 16)
        self.fire_rate = fire_rate

    @staticmethod
    def load(path: Path = WEIGHTS_PATH) -> "NCA":
        d = np.load(path)
        return NCA(d["W1"], d["b1"], d["W2"], float(d["fire_rate"]))

    def step(
        self,
        x: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        update_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Advance one step. Pass `update_mask` ((H, W) bool) for a
        deterministic step, otherwise it's sampled from `rng`."""
        pre = alive_mask(x)
        y = perceive(x)
        h = np.maximum(y @ self.W1 + self.b1, 0.0)
        dx = h @ self.W2
        if update_mask is None:
            rng = rng if rng is not None else np.random.default_rng()
            update_mask = rng.random(x.shape[:2]) <= self.fire_rate
        x = x + dx * update_mask[..., None].astype(np.float32)
        post = alive_mask(x)
        return (x * (pre & post)[..., None]).astype(np.float32)


def seed_state(h: int, w: int) -> np.ndarray:
    """All zeros except one center cell with alpha + hidden channels = 1."""
    x = np.zeros((h, w, CHANNELS), dtype=np.float32)
    x[h // 2, w // 2, 3:] = 1.0
    return x


def to_rgb(x: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """Composite premultiplied RGBA state over a background color -> uint8 RGB."""
    rgb = np.clip(x[..., :3], 0.0, 1.0)
    a = np.clip(x[..., 3:4], 0.0, 1.0)
    out = rgb + (1.0 - a) * (bg.astype(np.float32) / 255.0)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
