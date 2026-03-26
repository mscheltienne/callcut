from callcut._version import __version__
from callcut.utils._fixes import _preload_nvidia_npp
from callcut.utils.config import sys_info
from callcut.utils.logs import add_file_handler, set_log_level

_preload_nvidia_npp()

from callcut import (  # noqa: E402
    evaluation,
    extractors,
    io,
    nn,
    pipeline,
    training,
    utils,
)
