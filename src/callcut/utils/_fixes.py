"""Temporary bug-fixes awaiting an upstream fix."""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


# https://github.com/sphinx-gallery/sphinx-gallery/issues/1112
class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest, sphinx-gallery) work
    properly.
    """

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        # Even more ridiculous than this class, this must be sys.stdout (not
        # just stdout) in order for this to work (tested on OSX and Linux).
        if hasattr(sys.stdout, name):
            return getattr(sys.stdout, name)
        else:
            raise AttributeError(f"'file' object has not attribute '{name}'.")


def _preload_nvidia_npp() -> None:
    """Preload ``libnppicc`` from the ``nvidia-npp`` PyPI package.

    ``torchcodec`` links against ``libnppicc`` from the NVIDIA NPP (NVIDIA
    Performance Primitives) library. Contrary to ``torch`` which moved its packaging to
    depend on the ``nvidia-*`` PyPI packages and preloads them at import time,
    ``torchcodec`` neither declares ``nvidia-npp`` as a dependency nor preloads it,
    yielding a ``RuntimeError`` on import when the CUDA toolkit is not installed
    system-wide.

    The ``nvidia-npp`` PyPI package installs the ``.so`` files under
    ``site-packages/nvidia/cu*/lib/`` which is not on the default linker search path.
    This function preloads ``libnppicc`` globally so the dynamic linker can find it when
    ``torchcodec`` loads its native libraries.

    See Also
    --------
    https://github.com/meta-pytorch/torchcodec/issues/1309

    Notes
    -----
    This function must be called **before** any ``import torchcodec`` statement. It is
    safe to call multiple times (subsequent calls are no-ops). On non-Linux platforms,
    this function is a no-op (macOS torchcodec builds do not link against CUDA, and
    Windows does not ship torchcodec).
    """
    if sys.platform != "linux":
        return
    if getattr(preload_nvidia_npp, "_done", False):
        return
    for path in sys.path:
        if "site-packages" not in path:
            continue
        matches = glob.glob(
            os.path.join(path, "nvidia", "**", "lib", "libnppicc.so.*"),
            recursive=True,
        )
        if matches:
            try:
                ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break
    preload_nvidia_npp._done = True  # type: ignore[attr-defined]
