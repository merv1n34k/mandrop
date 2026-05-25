"""Async movie recorder for the live matplotlib display.

Grabs the already-rendered RGBA canvas buffer (no extra `draw()`) and pipes
raw frames into a single ffmpeg subprocess over stdin. A daemon thread
drains the queue, so the simulation thread never blocks on encoding or I/O.
Per-frame cost on the main thread is one memcpy of the canvas buffer plus
a `queue.put`.

Frame rate is decoupled from sim rate: every captured chunk becomes one
video frame at FPS (24). If encoding falls behind, the queue fills and new
frames are dropped rather than blocking the simulation.
"""

import queue
import subprocess
import threading
from datetime import datetime

import numpy as np


class MovieRecorder:
    FPS = 24
    CRF = 18

    def __init__(self, path=None, queue_size=16):
        self.path = path or f"mandrop_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.q = queue.Queue(maxsize=queue_size)
        self.proc = None
        self.worker = None
        self.dims = None
        self.dropped = 0
        self.written = 0
        self.failed = False

    def _start(self, w, h):
        w -= w & 1
        h -= h & 1
        self.dims = (w, h)
        try:
            self.proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "rgba",
                    "-s", f"{w}x{h}", "-r", str(self.FPS),
                    "-i", "-",
                    "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", str(self.CRF),
                    self.path,
                ],
                stdin=subprocess.PIPE,
            )
        except FileNotFoundError:
            print("movie: ffmpeg not found in PATH — recording disabled")
            self.failed = True
            return
        self.worker = threading.Thread(target=self._drain, daemon=True)
        self.worker.start()

    def _drain(self):
        while True:
            buf = self.q.get()
            if buf is None:
                return
            try:
                self.proc.stdin.write(buf)
                self.written += 1
            except BrokenPipeError:
                return

    def capture(self, fig):
        if self.failed:
            return
        rgba = np.asarray(fig.canvas.buffer_rgba())
        h, w, _ = rgba.shape
        if self.proc is None:
            self._start(w, h)
            if self.failed:
                return
        cw, ch = self.dims
        if (w, h) != (cw, ch):
            rgba = rgba[:ch, :cw]
        try:
            self.q.put(rgba.tobytes(), block=False)
        except queue.Full:
            self.dropped += 1

    def close(self):
        if self.worker is None:
            return
        self.q.put(None)
        self.worker.join()
        if self.proc is not None and self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.wait()
        print(f"movie → {self.path}  ({self.written} frames, {self.dropped} dropped)")
