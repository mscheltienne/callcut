I/O
---

.. currentmodule:: callcut.io

``callcut.io`` provides utilities for loading audio files and their annotations,
converting annotations to frame labels, and building PyTorch datasets for training.
The audio loading relies on ``torchaudio`` which under-the-hood delegates to
``torchcodec`` and ``ffmpeg``.

.. important::

    ``ffmpeg`` must be installed on your system to use ``callcut.io``.

Audio and Annotation Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated/api

    load_audio
    load_annotations

Recording Metadata
~~~~~~~~~~~~~~~~~~

Scan and validate recordings before building datasets. This pre-validation step
ensures all recordings have valid annotations and computes metadata (duration,
annotation count) without loading the full audio.

.. autosummary::
    :toctree: ../generated/api

    scan_recordings
    RecordingInfo

Label Generation
~~~~~~~~~~~~~~~~

Convert annotation intervals to per-frame binary labels for training.

.. autosummary::
    :toctree: ../generated/api

    intervals_to_frame_labels

Dataset
~~~~~~~

PyTorch Dataset for training call detection models.

.. autosummary::
    :toctree: ../generated/api

    CallDataset
