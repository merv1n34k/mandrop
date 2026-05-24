"""Save matplotlib figure frames to disk and assemble into an MP4 via ffmpeg.

Defaults are locked — no tuning knobs exposed at call sites:
- DPI: 300
- FPS: 24
- CRF: 18 (visually lossless)
- Codec: libx264 + yuv420p (browser / QuickTime compatible)

Usage in __main__.py is gated by a `--movie` flag.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


DPI    = 150   # 150 dpi at figsize=(20,12) → 3000×1800 PNG, ~4× faster than 300
FPS    = 24
CRF    = 18


class FrameSaver:
    """Saves a matplotlib Figure as PNG, named by step number.

    If `out_dir` is None, all `snap()` calls are no-ops — convenient for
    unconditional placement in the chunk callback.
    """

    def __init__(self, out_dir: str | Path | None):
        self._count = 0
        if out_dir is None:
            self.out_dir: Path | None = None
            return
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def is_active(self) -> bool:
        return self.out_dir is not None

    def snap(self, fig, step_num: int) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / f"frame_{step_num:09d}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


def frames_to_mp4(frames_dir: str | Path, out_path: str | Path) -> Path:
    """Assemble PNG sequence into MP4 via ffmpeg, with locked defaults.

    Uses ffmpeg's glob pattern matcher so frame numbers don't have to be
    contiguous from 0 (FrameSaver names by step_num, e.g. frame_..._000000200.png).
    Prints ffmpeg stderr on failure for diagnosability.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not on PATH. Install via `brew install ffmpeg` (macOS) "
            "or apt-get on Linux."
        )
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-pattern_type", "glob",
        "-i", str(frames_dir / "frame_*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        # libx264 requires even dimensions; round down per dimension.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-crf", str(CRF),
        "-preset", "medium",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Surface the real error rather than swallowing it
        raise RuntimeError(
            f"ffmpeg exit {proc.returncode}\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stderr:\n{proc.stderr}"
        )
    return out_path
