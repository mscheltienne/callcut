I/O
---

.. currentmodule:: callcut.io

``callcut.io`` provides utility to load audio files and their annotations. The audio
loading relies on ``torchaudio`` which under-the-hood delegates to ``torchcodec`` and
``ffmpeg``.

.. important::

    ``ffmpeg`` must be installed on your system to use ``callcut.io``.

.. autosummary::
    :toctree: ../generated/api

    load_audio
    load_annotations
