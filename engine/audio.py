"""Procedural soundtrack, synthesised from the Film timeline with numpy.

The music follows the picture in two ways:
- Scene.music picks a mood for each scene (chord progression, which
  instruments play, how busy the arpeggio and drums are). Instrument gains
  glide between moods, so scene changes never jump.
- Frame.sfx fires one-shot sounds exactly on the frame where something
  happens (a cell is born, a title locks in, the lizard is bitten ...).
  Pitched effects (pop, chime, blip) are snapped to the current chord so
  they stay musical.

MOODS: title, calm, curious, build, chaos, wonder, epic, tense, outro
SFX:   blip(pitch 0-1), pop(pitch 0-1), tick, chime(pitch 0-1), hit, whoosh,
       swish, rise(seconds), boom, glitch
"""
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

SR = 44100
BPM = 92.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SFX = ("blip", "pop", "tick", "chime", "hit", "whoosh", "swish", "rise", "boom", "glitch")
SFX_GAIN = 2.2  # sound effects vs. music (~ +7 dB over the original single-bus mix)

# chord name -> (bass root midi, pad voicing midi)
CHORDS: Dict[str, Tuple[int, Sequence[int]]] = {
    "Am9": (33, (57, 60, 64, 67, 71)),
    "Am": (33, (57, 60, 64, 69)),
    "Fmaj7": (29, (53, 57, 60, 64)),
    "F": (29, (53, 57, 60, 65)),
    "Cmaj7": (36, (55, 59, 60, 64)),
    "C": (36, (55, 60, 64, 67)),
    "G": (31, (55, 59, 62, 67)),
    "G6": (31, (55, 59, 62, 64)),
    "Em7": (28, (55, 59, 62, 64)),
    "Dm9": (38, (53, 57, 60, 64, 62)),
    "Bbmaj7": (34, (53, 57, 62, 65)),
    "E": (28, (56, 59, 64, 68)),
    "Esus": (28, (57, 59, 64, 69)),
}


@dataclass
class Mood:
    prog: Sequence[str]
    pad: float
    arp: float
    arp_div: int      # notes per beat
    bass: float
    bell: float
    drums: float      # 0 = none, 0.3 = hats only, 0.6 = + kick, 0.9 = + snare
    bright: float     # pad brightness (harmonic rolloff)


MOODS: Dict[str, Mood] = {
    "title": Mood(("Am9", "Fmaj7", "Cmaj7", "G6"), 0.9, 0.45, 2, 0.35, 0.5, 0.0, 0.5),
    "calm": Mood(("Am9", "Fmaj7", "Cmaj7", "Em7"), 0.8, 0.35, 2, 0.3, 0.25, 0.0, 0.35),
    "curious": Mood(("Dm9", "G6", "Cmaj7", "Am9"), 0.6, 0.5, 2, 0.4, 0.3, 0.3, 0.45),
    "build": Mood(("Am", "F", "C", "G"), 0.65, 0.55, 4, 0.55, 0.2, 0.6, 0.6),
    "chaos": Mood(("Am", "Bbmaj7", "Am", "E"), 0.55, 0.6, 4, 0.65, 0.1, 0.75, 0.7),
    "wonder": Mood(("Fmaj7", "G6", "Em7", "Am9"), 0.85, 0.35, 2, 0.3, 0.6, 0.2, 0.55),
    "epic": Mood(("Am", "F", "C", "G"), 0.9, 0.6, 4, 0.8, 0.35, 0.95, 0.8),
    "tense": Mood(("Am", "Esus", "Am", "E"), 0.55, 0.3, 2, 0.55, 0.1, 0.35, 0.3),
    "outro": Mood(("Fmaj7", "Cmaj7", "G6", "Am9"), 0.85, 0.3, 2, 0.3, 0.55, 0.0, 0.45),
}


def midi_hz(m) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69) / 12)


def _env(n: int, a: float, r: float) -> np.ndarray:
    """Attack (s) then exponential-ish release to the end."""
    t = np.arange(n) / SR
    att = np.clip(t / max(a, 1e-4), 0, 1)
    rel = np.clip((n / SR - t) / max(r, 1e-4), 0, 1)
    return att * rel


class Mix:
    def __init__(self, seconds: float):
        n = int(seconds * SR) + 4 * SR
        self.dry = np.zeros((n, 2), dtype=np.float64)
        self.wet = np.zeros((n, 2), dtype=np.float64)   # reverb send
        self.duck = np.ones(n, dtype=np.float64)

    def add(self, t0: float, sig: np.ndarray, gain: float = 1.0, pan: float = 0.0, send: float = 0.3) -> None:
        i0 = int(t0 * SR)
        if i0 >= len(self.dry) or len(sig) == 0:
            return
        sig = sig[: len(self.dry) - i0] * gain
        if sig.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            st = np.stack([sig * l, sig * r], axis=1) * np.sqrt(2)
        else:
            st = sig
        self.dry[i0: i0 + len(st)] += st * (1 - send * 0.5)
        self.wet[i0: i0 + len(st)] += st * send


# --- instruments ----------------------------------------------------------------

def pad_note(freq: float, dur: float, bright: float, rng) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for ch, det in enumerate((-0.12, 0.12)):
        f = freq * 2 ** (det / 12)
        ph = rng.random() * 2 * np.pi
        s = np.zeros(n)
        for h in range(1, 7):
            s += np.sin(2 * np.pi * f * h * t + ph * h) * (bright ** (h - 1)) / h
        # slow chorus wobble
        s *= 1 + 0.08 * np.sin(2 * np.pi * (0.3 + 0.1 * ch) * t)
        out[:, ch] = s
    return out * _env(n, 0.9, 1.4)[:, None] * 0.11


def pluck(freq: float, dur: float = 0.45, tone: float = 0.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + tone * 0.5 * np.sin(4 * np.pi * freq * t) + tone * 0.2 * np.sin(6 * np.pi * freq * t)
    return s * np.exp(-t * 9) * np.clip(t / 0.004, 0, 1) * 0.8


def bell(freq: float, dur: float = 2.5, index: float = 2.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    mod = np.sin(2 * np.pi * freq * 3.5 * t) * index * np.exp(-t * 3)
    return np.sin(2 * np.pi * freq * t + mod) * np.exp(-t * 2.2) * np.clip(t / 0.003, 0, 1) * 0.9


def bass_note(freq: float, dur: float) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
    s = np.tanh(1.6 * s)
    return s * _env(n, 0.01, 0.25) * 0.26


def kick() -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    f = 45 + 90 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * 0.5


def hat(rng, dur: float = 0.05) -> np.ndarray:
    n = int(dur * SR)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)
    return x * np.exp(-np.arange(n) / SR * 70) * 0.05


def snare(rng) -> np.ndarray:
    n = int(0.2 * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n) * np.exp(-t * 22) * 0.22
    x += np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30) * 0.25
    return x


def noise_sweep(rng, dur: float, f0: float, f1: float, rise: bool = True) -> np.ndarray:
    """Band-limited noise whose one-pole low-pass cutoff sweeps f0 -> f1."""
    n = int(dur * SR)
    x = rng.standard_normal(n)
    fc = np.geomspace(f0, f1, n)
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.zeros(n)
    acc = 0.0
    for i in range(0, n, 64):  # piecewise-constant coefficients, vectorised per block
        blk = x[i: i + 64]
        ai = a[i]
        out = np.empty(len(blk))
        for j, v in enumerate(blk):
            acc = ai * acc + (1 - ai) * v
            out[j] = acc
        y[i: i + 64] = out
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** (0.7 if rise else 2.0)
    return y * env


# --- reverb ------------------------------------------------------------------------

def reverb(x: np.ndarray, seconds: float = 2.6, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_ir = int(seconds * SR)
    t = np.arange(n_ir) / SR
    ir = rng.standard_normal((n_ir, 2)) * np.exp(-t * 6.9 / seconds)[:, None]
    ir[: int(0.012 * SR)] = 0  # pre-delay
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    out = np.zeros((len(x) + n_ir, 2))
    block = 1 << 18
    nfft = 1 << int(np.ceil(np.log2(block + n_ir)))
    IR = np.fft.rfft(ir, nfft, axis=0)
    for i in range(0, len(x), block):
        seg = x[i: i + block]
        y = np.fft.irfft(np.fft.rfft(seg, nfft, axis=0) * IR, nfft, axis=0)[: len(seg) + n_ir]
        out[i: i + len(y)] += y
    return out[: len(x)]


# --- composition --------------------------------------------------------------------

def _mood_at(film, t: float) -> str:
    from .timeline import TICK_FPS
    s = film.scene_at(int(t * TICK_FPS))
    return s.music if s.music in MOODS else "calm"


def _chord_at(film, bar: int) -> Tuple[int, Sequence[int]]:
    m = MOODS[_mood_at(film, bar * BAR + 0.1)]
    return CHORDS[m.prog[bar % len(m.prog)]]


def _snap(chord: Sequence[int], p: float, octave_up: int = 1) -> float:
    """Map p in [0, 1] to a chord tone across two octaves."""
    tones = sorted(set(chord))
    ladder = [n + 12 * o for o in range(octave_up, octave_up + 2) for n in tones]
    return float(midi_hz(ladder[int(np.clip(p, 0, 0.999) * len(ladder))]))


def synthesize(film, seed: int = 7, normalize: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = film.duration
    mix = Mix(T)
    n_bars = int(np.ceil(T / BAR)) + 1

    # per-bar layer gains, smoothed so moods glide
    names = ("pad", "arp", "bass", "bell", "drums", "bright")
    g = {k: np.array([getattr(MOODS[_mood_at(film, b * BAR + BAR / 2)], k) for b in range(n_bars)])
         for k in names}
    for k in names:
        v = g[k].copy()
        g[k] = np.convolve(np.pad(v, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")

    for b in range(n_bars):
        t0 = b * BAR
        if t0 >= T:
            break
        root, voicing = _chord_at(film, b)
        mood = MOODS[_mood_at(film, t0 + BAR / 2)]
        # pad: whole bar + overlap tail
        if g["pad"][b] > 0.02:
            for k, m in enumerate(voicing):
                mix.add(t0, pad_note(float(midi_hz(m)), BAR + 1.2, 0.35 + 0.4 * g["bright"][b], rng),
                        gain=g["pad"][b], send=0.55)
        # bass
        if g["bass"][b] > 0.02:
            pulses = [0, 2] if mood.drums < 0.6 else [0, 1.5, 2, 3.5]
            for pb in pulses:
                mix.add(t0 + pb * BEAT, bass_note(float(midi_hz(root + 12)), BEAT * 1.4),
                        gain=g["bass"][b], send=0.1)
        # arpeggio: up-down over chord tones, 2 octaves
        if g["arp"][b] > 0.02:
            tones = sorted(set(voicing))
            ladder = [n + 12 for n in tones] + [n + 24 for n in tones[:2]]
            pat = ladder + ladder[-2:0:-1]
            div = mood.arp_div
            for k in range(4 * div):
                note = pat[(b * 4 * div + k) % len(pat)]
                accent = 1.0 if k % div == 0 else 0.7
                if div == 4 and rng.random() < 0.15:
                    continue
                mix.add(t0 + k * BEAT / div, pluck(float(midi_hz(note)), 0.5, 0.4 + 0.3 * g["bright"][b]),
                        gain=g["arp"][b] * accent * (0.8 if div == 4 else 1.0),
                        pan=0.5 * np.sin(k * 0.7), send=0.45)
                # ping-pong echo
                mix.add(t0 + k * BEAT / div + 0.75 * BEAT, pluck(float(midi_hz(note)), 0.5, 0.3),
                        gain=g["arp"][b] * accent * 0.28, pan=-0.6 * np.sin(k * 0.7), send=0.6)
        # bells: a sparse melody on chord tones
        if g["bell"][b] > 0.02 and b % 2 == 0:
            tones = sorted(set(voicing))
            for k, beat in enumerate((0, 2.5) if g["bell"][b] > 0.4 else (0,)):
                m = tones[(b // 2 + k * 2) % len(tones)] + 24
                mix.add(t0 + beat * BEAT, bell(float(midi_hz(m))), gain=g["bell"][b] * 0.7,
                        pan=0.3 * (-1) ** k, send=0.65)
        # drums
        d = g["drums"][b]
        if d > 0.15:
            for k in range(8):
                mix.add(t0 + k * BEAT / 2, hat(rng), gain=d * (1.0 if k % 2 else 0.6), pan=0.25, send=0.1)
        if d > 0.45:
            for kb in ((0, 2) if d < 0.8 else (0, 1.5, 2, 2.75)):
                mix.add(t0 + kb * BEAT, kick(), gain=d * 0.9, send=0.05)
        if d > 0.7:
            for sb in (1, 3):
                mix.add(t0 + sb * BEAT, snare(rng), gain=d * 0.6, send=0.35)

    # --- sound effects (own bus, so their level vs. the music is one knob) ---
    from .timeline import TICK_FPS
    music, mix = mix, Mix(T)
    last: Dict[str, float] = {}
    min_gap = {"tick": 0.07, "pop": 0.06, "blip": 0.06, "chime": 0.15, "whoosh": 0.4, "swish": 0.2}
    for tick, name, param in film.sfx:
        t = tick / TICK_FPS
        if t - last.get(name, -9) < min_gap.get(name, 0.12):
            continue
        last[name] = t
        chord = _chord_at(film, int(t // BAR))[1]
        if name == "blip":
            mix.add(t, pluck(_snap(chord, param, 2), 0.18, 0.2), gain=0.3, send=0.3, pan=0.2)
        elif name == "pop":
            mix.add(t, pluck(_snap(chord, param, 1), 0.3, 0.6), gain=0.35, send=0.35,
                    pan=float(np.clip(param * 2 - 1, -0.7, 0.7)))
        elif name == "tick":
            n = int(0.03 * SR)
            s = np.sin(2 * np.pi * 2200 * np.arange(n) / SR) * np.exp(-np.arange(n) / SR * 180) * 0.25
            mix.add(t, s, gain=0.5, send=0.15)
        elif name == "chime":
            mix.add(t, bell(_snap(chord, param, 2), 3.0, 1.5), gain=0.45, send=0.7)
        elif name == "hit":
            mix.add(t, kick(), gain=1.0, send=0.3)
            mix.add(t, noise_sweep(rng, 0.5, 6000, 300, rise=False), gain=0.35, send=0.6)
            mix.add(t, bass_note(float(midi_hz(_chord_at(film, int(t // BAR))[0])), 1.4), gain=0.8, send=0.3)
            i0 = int(t * SR)
            k = np.linspace(0.55, 1, int(0.6 * SR))
            music.duck[i0: i0 + len(k)] = np.minimum(music.duck[i0: i0 + len(k)], k[: len(music.duck) - i0])
        elif name == "whoosh":
            mix.add(max(0, t - 0.25), noise_sweep(rng, 0.8, 300, 5000), gain=0.22, send=0.5)
        elif name == "swish":
            mix.add(t, noise_sweep(rng, 0.35, 1500, 9000), gain=0.16, send=0.3, pan=0.3)
        elif name == "rise":
            dur = max(0.5, param)
            n = int(dur * SR)
            tt = np.arange(n) / SR
            f = 180 * 2 ** (tt / dur * 2.5)
            s = np.sin(2 * np.pi * np.cumsum(f) / SR) * (tt / dur) ** 2 * 0.12
            mix.add(t, s, gain=1.0, send=0.5)
            mix.add(t, noise_sweep(rng, dur, 200, 8000), gain=0.18, send=0.5)
        elif name == "boom":
            n = int(2.0 * SR)
            tt = np.arange(n) / SR
            mix.add(t, np.sin(2 * np.pi * (38 + 30 * np.exp(-tt * 4)) * tt) * np.exp(-tt * 1.8) * 0.9,
                    gain=1.0, send=0.4)
        elif name == "glitch":
            n = int(0.12 * SR)
            sq = np.sign(np.sin(2 * np.pi * rng.choice([220, 330, 440, 660]) * np.arange(n) / SR))
            mix.add(t, sq * (rng.random(n) < 0.7) * 0.08, gain=1.0, send=0.2)

    fx = mix
    dry = music.dry * music.duck[:, None] + SFX_GAIN * fx.dry
    out = dry + reverb(music.wet + SFX_GAIN * fx.wet) * 0.9
    n = int(T * SR)
    out = out[:n]
    # master: gentle fade in/out, soft clip, loudness normalise
    fade_in = np.clip(np.arange(n) / (0.8 * SR), 0, 1)
    fade_out = np.clip((n - np.arange(n)) / (3.0 * SR), 0, 1)
    out *= (fade_in * fade_out)[:, None]
    if not normalize:
        return out.astype(np.float32)
    rms = np.sqrt(np.mean(out ** 2)) + 1e-9
    out *= 0.12 / rms
    out = np.tanh(out * 1.2) / np.tanh(1.2)
    return (out * 0.89).astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)


def soundtrack_for(film, film_path: Path) -> Tuple[Optional[Path], Optional[np.ndarray], int]:
    """Soundtrack cached next to the film cache (same key)."""
    wav = film_path.with_suffix(".wav")
    if not wav.exists():
        write_wav(wav, synthesize(film))
    return wav, read_wav(wav), SR
