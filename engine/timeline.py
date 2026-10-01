"""Film timeline: runs every scene once, applies transitions, and stores the
result as a compact, seekable sequence of frames at a fixed tick rate.

Everything downstream (terminal player with seeking, mp4 renderer, soundtrack
synthesis, .srt export) reads this `Film` instead of running the generators,
so they all agree tick-for-tick. Films are cached on disk, keyed by a hash of
the source code, so the (slow) simulation pass runs once.
"""
import hashlib
import json
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .frame import Frame, Scene
from .fx import transition

TICK_FPS = 24
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "build" / "cache"


def _pack(f: Frame) -> bytes:
    flags = f.active.astype(np.uint8) | (f.bg_active.astype(np.uint8) << 1)
    return zlib.compress(
        np.ascontiguousarray(f.chars).view(np.uint32).tobytes()
        + f.colors.tobytes() + f.bg.tobytes() + flags.tobytes(),
        1,
    )


def _unpack(blob: bytes, rows: int, cols: int) -> Frame:
    raw = zlib.decompress(blob)
    n = rows * cols
    chars = np.frombuffer(raw, dtype=np.uint32, count=n).view("<U1").reshape(rows, cols).copy()
    o = 4 * n
    colors = np.frombuffer(raw, dtype=np.uint8, count=3 * n, offset=o).reshape(rows, cols, 3).copy()
    o += 3 * n
    bg = np.frombuffer(raw, dtype=np.uint8, count=3 * n, offset=o).reshape(rows, cols, 3).copy()
    o += 3 * n
    flags = np.frombuffer(raw, dtype=np.uint8, count=n, offset=o).reshape(rows, cols)
    return Frame(chars=chars, colors=colors, active=(flags & 1).astype(bool), bg=bg,
                 bg_active=(flags & 2).astype(bool))


@dataclass
class SceneSpan:
    name: str
    start: int   # first tick
    end: int     # one past the last tick
    music: str
    transition: str       # declared entry transition
    ticks: int = 0        # declared transition length
    applied: bool = True  # False for a film's first scene (nothing to transition from yet)
    chapter: str = ""     # chapter indicator shown in a corner ("" = none, e.g. on title cards)
    chapter_en: str = ""


@dataclass
class Film:
    rows: int
    cols: int
    blobs: List[bytes] = field(default_factory=list)
    subs: List[Tuple[str, str]] = field(default_factory=list)     # per unique frame
    tick_frame: List[int] = field(default_factory=list)           # tick -> frame index
    sfx: List[Tuple[int, str, float]] = field(default_factory=list)
    scenes: List[SceneSpan] = field(default_factory=list)
    _lru: "OrderedDict[int, Frame]" = field(default_factory=OrderedDict, repr=False)

    @property
    def n_ticks(self) -> int:
        return len(self.tick_frame)

    @property
    def duration(self) -> float:
        return self.n_ticks / TICK_FPS

    def frame_index(self, tick: int) -> int:
        return self.tick_frame[max(0, min(tick, self.n_ticks - 1))]

    def frame(self, tick: int) -> Frame:
        i = self.frame_index(tick)
        f = self._lru.get(i)
        if f is None:
            f = _unpack(self.blobs[i], self.rows, self.cols)
            f.subtitle_zh, f.subtitle_en = self.subs[i]
            self._lru[i] = f
            if len(self._lru) > 64:
                self._lru.popitem(last=False)
        else:
            self._lru.move_to_end(i)
        return f

    def chapter_at(self, tick: int) -> Tuple[str, str]:
        s = self.scene_at(max(0, min(tick, self.n_ticks - 1)))
        return s.chapter, s.chapter_en

    def subtitle(self, tick: int) -> Tuple[str, str]:
        return self.subs[self.frame_index(tick)]

    def scene_at(self, tick: int) -> SceneSpan:
        for s in self.scenes:
            if s.start <= tick < s.end:
                return s
        return self.scenes[-1]

    def cues(self) -> List[Tuple[float, float, str, str]]:
        """Subtitle cues (start_sec, end_sec, zh, en) for .srt export."""
        out, cur, start = [], ("", ""), 0
        for t in range(self.n_ticks + 1):
            s = self.subtitle(t) if t < self.n_ticks else ("", "")
            if s != cur:
                if any(cur):
                    out.append((start / TICK_FPS, t / TICK_FPS, cur[0], cur[1]))
                cur, start = s, t
        return out

    # --- persistence ---
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lens = np.array([len(b) for b in self.blobs], dtype=np.int64)
        meta = {
            "rows": self.rows, "cols": self.cols, "subs": self.subs, "sfx": self.sfx,
            "scenes": [s.__dict__ for s in self.scenes],
        }
        import os
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, data=np.frombuffer(b"".join(self.blobs), dtype=np.uint8), lens=lens,
                 tick_frame=np.array(self.tick_frame, dtype=np.int32),
                 meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
        tmp.replace(path)

    @staticmethod
    def load(path: Path) -> "Film":
        d = np.load(path)
        meta = json.loads(d["meta"].tobytes().decode())
        data = d["data"].tobytes()
        offs = np.concatenate([[0], np.cumsum(d["lens"])])
        film = Film(rows=meta["rows"], cols=meta["cols"])
        film.blobs = [data[offs[i]: offs[i + 1]] for i in range(len(offs) - 1)]
        film.subs = [tuple(s) for s in meta["subs"]]
        film.tick_frame = d["tick_frame"].tolist()
        film.sfx = [tuple(s) for s in meta["sfx"]]
        film.scenes = [SceneSpan(**s) for s in meta["scenes"]]
        return film


def build_film(scenes: Sequence[Scene], rows: int, cols: int,
               progress: Optional[Callable[[str, int], None]] = None) -> Film:
    """Run all scenes. `progress(scene_name, ticks_so_far)` is called per frame."""
    film = Film(rows=rows, cols=cols)
    last_blob: Optional[bytes] = None
    prev_last: Optional[Frame] = None
    prev_name = ""
    for si, scene in enumerate(scenes):
        if TICK_FPS % scene.fps:
            raise ValueError(f"scene {scene.name}: fps {scene.fps} must divide {TICK_FPS}")
        per = TICK_FPS // scene.fps
        start = film.n_ticks
        trans = scene.transition if prev_last is not None else "cut"
        if prev_name.endswith("-title") and trans != "cut":
            trans = "fade"   # after a chapter title card: a plain fade through black, no glyph effects
        t_ticks = scene.transition_ticks if trans != "cut" else 0
        if trans != "cut":
            film.sfx.append((start, "whoosh", 1.0))
        last: Optional[Frame] = None
        for frame in scene.frame_gen():
            if frame.chars.shape != (rows, cols):
                raise ValueError(f"scene {scene.name}: frame shape {frame.chars.shape} != {(rows, cols)}")
            for name, param in frame.sfx:
                film.sfx.append((film.n_ticks, name, float(param)))
            for k in range(per):
                tick_in_scene = film.n_ticks - start
                if tick_in_scene < t_ticks:
                    out = transition(trans, prev_last, frame, (tick_in_scene + 1) / (t_ticks + 1),
                                     tick=film.n_ticks, seed=si)
                    out.subtitle_zh, out.subtitle_en = frame.subtitle_zh, frame.subtitle_en
                else:
                    out = frame
                if k == 0 or tick_in_scene <= t_ticks:
                    blob = _pack(out)
                    if blob != last_blob or not film.subs or film.subs[-1] != (out.subtitle_zh, out.subtitle_en):
                        film.blobs.append(blob)
                        film.subs.append((out.subtitle_zh, out.subtitle_en))
                        last_blob = blob
                film.tick_frame.append(len(film.blobs) - 1)
            last = frame
            if progress:
                progress(scene.name, film.n_ticks)
        if last is None:
            raise ValueError(f"scene {scene.name} yielded no frames")
        prev_last = last.copy()
        prev_name = scene.name
        # a film's first scene keeps its *declared* entry transition: stitch()
        # applies it over the previous chapter's last frame
        film.scenes.append(SceneSpan(scene.name, start, film.n_ticks, scene.music,
                                     trans if si > 0 else scene.transition,
                                     scene.transition_ticks, si > 0))
    return film


def stitch(films: Sequence[Film]) -> Film:
    """Concatenate films; each later film's first scene gets the entry
    transition it declared, blended over the previous film's last frame."""
    out = Film(rows=films[0].rows, cols=films[0].cols)
    for fi, f in enumerate(films):
        t_off, b_off = out.n_ticks, len(out.blobs)
        prev_last = out.frame(out.n_ticks - 1).copy() if fi > 0 else None
        out.blobs.extend(f.blobs)
        out.subs.extend(f.subs)
        out.tick_frame.extend(i + b_off for i in f.tick_frame)
        out.sfx.extend((t + t_off, n, p) for t, n, p in f.sfx)
        for k, s in enumerate(f.scenes):
            out.scenes.append(SceneSpan(s.name, s.start + t_off, s.end + t_off, s.music, s.transition,
                                        s.ticks, s.applied or (k == 0 and fi > 0), s.chapter, s.chapter_en))
        first = f.scenes[0]
        if prev_last is not None and first.transition != "cut" and first.ticks > 0:
            out.sfx.append((t_off, "whoosh", 1.0))
            n = min(first.ticks, first.end - first.start)
            for k in range(n):
                raw = f.frame(k)
                mixed = transition(first.transition, prev_last, raw, (k + 1) / (first.ticks + 1),
                                   tick=t_off + k, seed=fi * 101)
                out.blobs.append(_pack(mixed))
                out.subs.append((raw.subtitle_zh, raw.subtitle_en))
                out.tick_frame[t_off + k] = len(out.blobs) - 1
        out._lru.clear()
    return out


def _chapter_sources(module) -> List[Path]:
    """The chapter's own file, chapters/common.py, and any chapter module it imports."""
    import re
    own = Path(module.__file__).resolve()
    paths = [own, ROOT / "chapters" / "common.py"]
    for m in re.findall(r"(?:from|import)\s+(?:chapters\.|\.)(chapter\w+)", own.read_text()):
        p = ROOT / "chapters" / f"{m}.py"
        if p.exists():
            paths.append(p)
    return paths


def chapter_hash(module, extra: str = "") -> str:
    h = hashlib.sha1(extra.encode())
    files = []
    for sub in ("engine", "sim", "assets"):
        files += [p for p in sorted((ROOT / sub).rglob("*"))
                  if p.is_file() and p.suffix in (".py", ".npz", ".png", ".jpg", ".rule", ".rle", ".json", ".txt")]
    for p in files + _chapter_sources(module):
        h.update(str(p.relative_to(ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _build_chapter_job(args):
    key, module_name, rows, cols, path = args
    import importlib
    import time as _time
    t0 = _time.time()
    mod = importlib.import_module(module_name)
    film = build_film(mod.build_scenes(rows, cols), rows, cols)
    zh, en = getattr(mod, "CHAPTER_LABEL", ("", ""))
    skip = set(getattr(mod, "NO_CHAPTER_LABEL", ()))
    for sp in film.scenes:   # no corner label on title cards (they already say it)
        if not sp.name.endswith("-title") and sp.name != "ch1-opening" and sp.name not in skip:
            sp.chapter, sp.chapter_en = zh, en
    film.save(Path(path))
    return key, _time.time() - t0


def load_or_build_chapters(keys: Sequence[str], modules: dict, rows: int, cols: int,
                           use_cache: bool = True, workers: int = 0,
                           progress: Optional[Callable[[str, float, int, int], None]] = None
                           ) -> Tuple[Film, Path]:
    """Per-chapter cached films, built in parallel, stitched into one film."""
    import os
    from multiprocessing import get_context
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {k: CACHE_DIR / f"film-{k}-{chapter_hash(modules[k], k)}.npz" for k in keys}
    todo = [k for k in dict.fromkeys(keys) if not (use_cache and paths[k].exists())]
    if todo:
        jobs = [(k, modules[k].__name__, rows, cols, str(paths[k])) for k in todo]
        n = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
        done = 0
        if progress:
            progress("", 0.0, 0, len(jobs))
        with get_context("fork").Pool(n) as pool:
            for key, secs in pool.imap_unordered(_build_chapter_job, jobs):
                done += 1
                if progress:
                    progress(key, secs, done, len(jobs))
        for k in todo:
            for old in CACHE_DIR.glob(f"film-{k}-*"):
                if old.stem != paths[k].stem:
                    old.unlink(missing_ok=True)
    if len(keys) == 1:
        return Film.load(paths[keys[0]]), paths[keys[0]]
    combo = hashlib.sha1("|".join(paths[k].name for k in keys).encode()).hexdigest()[:16]
    tag = "all" if len(keys) > 3 else "+".join(keys)
    out_path = CACHE_DIR / f"film-{tag}-{combo}.npz"
    if out_path.exists():
        return Film.load(out_path), out_path
    film = stitch([Film.load(paths[k]) for k in keys])
    film.save(out_path)
    for old in CACHE_DIR.glob(f"film-{tag}-*"):
        if old.stem != out_path.stem:
            old.unlink(missing_ok=True)
    return film, out_path
