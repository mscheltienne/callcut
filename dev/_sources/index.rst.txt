.. include:: ./links.inc

**CallCut**
===========

.. toctree::
   :hidden:

   usage
   api/index
   changes/index

This is an automatic animal call cutter which will help you process your audio
recordings.

Installation
------------

``callcut`` relies on the `torch`_ ecosystem, thus audio-loading goes through
`torchaudio`_ and `torchcodec`_ which requires `ffmpeg`_.

.. dropdown:: FFmpeg installation

    For conda-users, you can use one of:

    .. code-block:: shell

        conda install "ffmpeg"
        conda install "ffmpeg" -c conda-forge

    Otherwise, if you prefer ``pip`` or `uv`_, `ffmpeg`_ must be available in your
    system ``PATH``:

    .. tab-set::

        .. tab-item:: Windows

            1. Download the ``"full-shared"`` build from https://www.gyan.dev/ffmpeg/builds/
            2. Extract somewhere (e.g., ``C:\ffmpeg``)
            3. Add the ``bin\`` folder to your system PATH

        .. tab-item:: macOS

            Install using `homebrew`_:

            .. code-block:: shell

                brew install ffmpeg

        .. tab-item:: Linux

            Install using your package manager, depending on your distro:

            .. code-block:: shell

                sudo apt install ffmpeg  # Ubuntu
                sudo dnf install ffmpeg  # Fedora

Beside this system dependency, ``callcut`` is a plain python package which can be
installed with ``pip`` (or `uv`_):

.. tab-set::

    .. tab-item:: PyPI

        ``callcut`` is available on PyPI:

        .. code-block:: shell

            pip install callcut

    .. tab-item:: Source

        To install from source, you can clone the repository then either use ``pip``:

        .. code-block:: shell

            pip install -e .

        Or `uv`_:

        .. code-block:: shell

            uv sync
