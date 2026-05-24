"""Save matplotlib figure frames to disk and assemble into an MP4 via ffmpeg.

Usage from `__main__`:

    from mandrop.movie import FrameSaver, frames_to_mp4

    saver = FrameSaver(out_dir="frames/run_2026-05-24")  # or None to disable
    ...
    # inside update_plots callback:
    saver.snap(fig, step_num)
    ...
    # at end:
    if saver.is_active():
        frames_to_mp4(saver.out_dir, "run.mp4", fps=30)

`FrameSaver` is a no-op stub when `out_dir is None`, so callers can wrap
it unconditionally without checks at every call site.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FrameSaver:
    """Saves a matplotlib Figure as PNG, named by step number.

    If `out_dir` is None, all `snap()` calls are no-ops — convenient for
    unconditional placement in the chunk callback.
    """

    def __init__(self, out_dir: str | Path | None, dpi: int = 100):
        self.dpi = dpi
        if out_dir is None:
            self.out_dir: Path | None = None
            self._count = 0
            return
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def is_active(self) -> bool:
        return self.out_dir is not None

    def snap(self, fig, step_num: int) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / f"frame_{step_num:09d}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count


def frames_to_mp4(
    frames_dir: str | Path,
    out_path: str | Path,
    fps: int = 30,
    crf: int = 23,
    pattern: str = "frame_%09d.png",
) -> Path:
    """Assemble a PNG sequence into MP4 via ffmpeg.

    Args:
        frames_dir: directory containing the PNG sequence.
        out_path:   output mp4 path.
        fps:        playback frames-per-second.
        crf:        x264 quality (18-28 typical; lower = bigger, higher quality).
        pattern:    printf-style filename pattern (default matches FrameSaver).

    Returns the output path on success. Raises FileNotFoundError if ffmpeg
    isn't on PATH, or subprocess.CalledProcessError if the encode fails.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install via `brew install ffmpeg` (macOS) "
            "or apt-get on Linux."
        )
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",   # required for browser/QuickTime playback
        "-crf", str(crf),
        "-preset", "medium",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path
