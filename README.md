[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![tests](https://github.com/mscheltienne/callcut/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/mscheltienne/callcut/actions/workflows/pytest.yaml)
[![doc](https://github.com/mscheltienne/callcut/actions/workflows/doc.yaml/badge.svg?branch=main)](https://github.com/mscheltienne/callcut/actions/workflows/doc.yaml)

# CallCut

## Installation

`callcut` relies on the `torch` ecosystem, thus audio-loading goes through `torchaudio`
and `torchcodec` which requires `ffmpeg`.

----------------------------------------------------------------------------------------

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

----------------------------------------------------------------------------------------

After that, the project can be installed as any python package distributed on PyPI:

```
pip install callcut
```

For installation from source using [uv](https://docs.astral.sh/uv/):

```
uv sync
```

To pull the torch binaries for your platform. Otherwise, you can pick either of the 2
extras with `pip install callcut[cpu]` or `pip install callcut[cu128]` which will
resolve to the PyPI `torch` dependencies- or alternatively you can install `callcut`
without any extra and manually install the `torch` dependencies from your preferred
index.

## Usage

`callcut` detects animal calls in audio recordings. Each audio file needs a
companion annotation CSV (e.g. `recording_annotations.csv`) with `start_seconds`
and `stop_seconds` columns (values in milliseconds).

### Training

```python
from pathlib import Path

import lightning as L

from callcut.evaluation import HysteresisDecoder, IoUMatcher
from callcut.extractors import SNRExtractor
from callcut.nn import TinySegCNN
from callcut.pipeline import evaluate_recordings, save_pipeline
from callcut.training import (
    BCEWithLogitsLoss,
    CallDataModule,
    CallDetectorModule,
    LoggingCallback,
    SaveBestModelCallback,
)

L.seed_everything(42)

wav_files = sorted(Path("data/").glob("*.wav"))

# Feature extraction: multi-band SNR
extractor = SNRExtractor(sample_rate=32_000)

# Data module: handles train/val/test splitting at the recording level
dm = CallDataModule(recordings=wav_files, extractor=extractor, num_workers=0)
dm.setup("fit")

# Model: lightweight 1D CNN
window_frames = extractor.seconds_to_frames(2.0)
model = TinySegCNN(n_bands=extractor.n_features, window_frames=window_frames)
module = CallDetectorModule(model, loss=BCEWithLogitsLoss())

# Train
trainer = L.Trainer(
    max_epochs=10,
    accelerator="cpu",
    devices=1,
    callbacks=[LoggingCallback(), SaveBestModelCallback("best_weights.pt")],
    enable_checkpointing=False,
)
trainer.fit(module, datamodule=dm)

# Save the full pipeline (model + extractor + decoder config)
decoder = HysteresisDecoder()
save_pipeline(model, extractor, decoder, "pipeline.pt")
```

### Evaluation

```python
from callcut.evaluation import HysteresisDecoder, IoUMatcher
from callcut.pipeline import evaluate_recordings, load_pipeline

model, extractor, decoder = load_pipeline("pipeline.pt")
# dm.test_recordings is available after dm.setup("fit")
report = evaluate_recordings(
    model, extractor, dm.test_recordings, decoder, IoUMatcher()
)
print(report)
```

### Inference on new recordings

```python
from pathlib import Path

from callcut.pipeline import load_pipeline, predict_recordings

model, extractor, decoder = load_pipeline("pipeline.pt")
audio_files = sorted(Path("new_data/").glob("*.wav"))
predictions = predict_recordings(model, extractor, audio_files, decoder)

for pred in predictions:
    print(f"{pred.audio_path.name}: {len(pred.intervals)} calls")
    for interval in pred.intervals:
        print(f"  {interval.onset:.3f}s - {interval.offset:.3f}s")
```
