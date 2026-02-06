[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![tests](https://github.com/mscheltienne/callcut/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/mscheltienne/callcut/actions/workflows/pytest.yaml)
[![doc](https://github.com/mscheltienne/callcut/actions/workflows/doc.yaml/badge.svg?branch=main)](https://github.com/mscheltienne/callcut/actions/workflows/doc.yaml)

# CallCut

## Installation

`callcut` relies on the `torch` ecosystem, thus audio-loading goes through `torchaudio`
and `torchcodec` which requires `ffmpeg`.

<details>
<summary>FFmpeg installation</summary>

For conda-users, you can use one of:

```shell
conda install "ffmpeg"
conda install "ffmpeg" -c conda-forge
```

But you can also use `uv` or `pip` and install `ffmpeg` as a system dependency:

On Linux:

```
sudo apt install ffmpeg  # Ubuntu
sudo dnf install ffmpeg  # Fedora
```

On macOS:

```shell
brew install ffmpeg
```

On Windows:

1. Download the `"full-shared"` build from https://www.gyan.dev/ffmpeg/builds/
2. Extract somewhere (e.g., `C:\ffmpeg`)
3. Add the `bin\` folder to your system PATH

</details>

After that, the project can be installed as any python package. For installation from
source using [uv](https://docs.astral.sh/uv/), you can use directly the keys:

```
uv sync --extra cpu
uv sync --extra cu128
```

To pull the torch binaries for your platform. Otherwise, you can pick either or the 2
extras with `pip install callcut[cpu]` or `pip install callcut[cu128]` which will
resolve to the PyPI `torch` dependencies- or alternatively you can install `callcut`
without any extra and manually install the `torch` dependencies from your preferred
index.
