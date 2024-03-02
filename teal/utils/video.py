from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Optional, Tuple, Union

import numpy as np


class VideoWriter:
    """
    Utility to write frames to a video file using ffmpeg.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: int = 25,
        qual=16,
    ):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg and make sure it is in your PATH."
            )

        self.ffmpeg_path = Path(ffmpeg_path)
        self.output_path = Path(output_path)
        self.proc: Optional[subprocess.Popen] = None
        self.image_size: Optional[Tuple[int, int]] = None  # (height, width)
        self.fps = fps
        self.quality = qual  # crf 0-51, 0 is lossless

    def write_frame(self, frame: np.ndarray):
        assert (
            frame.ndim == 3
        ), f"Expected frame to have 3 dimensions, got {frame.ndim} instead."

        if frame.shape[0] % 2 != 0:
            # ffmpeg requires even height
            frame = np.pad(frame, ((0, 1), (0, 0), (0, 0)))

        if self.proc is None:
            self.open(frame.shape[1], frame.shape[0])

        assert self.proc is not None, "Stream not open. "
        assert self.image_size is not None, "Image size not set. "
        assert frame.shape == (*self.image_size, 3), (
            f"Expected frame to have shape {(*self.image_size, 3)}, "
            f"got {frame.shape} instead."
        )
        assert (
            frame.dtype == np.uint8
        ), f"Expected frame to have dtype np.uint8, got {frame.dtype} instead."

        assert self.proc.stdin is not None, "Stream not open. "

        try:
            self.proc.stdin.write(frame.astype(np.uint8).tobytes())
        except:
            self._handle_error()

    def open(self, width: int, height: int):
        assert self.proc is None, "Stream already open. "
        self.image_size = (height, width)
        cmd = (
            f"{str(self.ffmpeg_path)} -y -f rawvideo -vcodec rawvideo -s {width}x{height} "
            f"-pix_fmt rgb24 -r {self.fps} -i - -threads 0 -preset fast -y -an "
            f"-pix_fmt yuv420p -crf {self.quality} {shlex.quote(str(self.output_path))}"
        )
        self.proc = subprocess.Popen(
            shlex.split(cmd), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def close(self):
        if self.proc is None:
            return
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()
        except:
            self._handle_error()
        finally:
            self.proc = None
            self.image_size = None

    def _handle_error(self):
        assert self.proc is not None, "Cannot handle error with no process. "

        if self.proc.stderr is not None:
            error_msg = self.proc.stderr.read()
            if error_msg:
                raise RuntimeError(
                    f"Error while writing video. FFMPEG exited with code {self.proc.returncode}. Error message:\n"
                    f"================\n"
                    f"{error_msg}"
                )

        raise RuntimeError(
            f"Error while writing video. FFMPEG exited with code {self.proc.returncode}."
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
