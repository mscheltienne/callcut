Inference
---------

.. currentmodule:: callcut.inference

``callcut.inference`` provides tools for converting frame-level probability outputs into
discrete call intervals.

.. note::

    For running inference on full recordings, use the :meth:`~callcut.nn.BaseDetector.predict`
    method on the model directly.

Decoding
~~~~~~~~

Decoders convert frame-level probabilities (values in ``[0, 1]`` for each time frame)
into a list of discrete call intervals with onset and offset times.

:class:`HysteresisDecoder` uses hysteresis thresholding with separate enter/exit
thresholds to avoid rapid on/off switching when probabilities hover near a single
threshold. It also merges nearby intervals and filters out short detections. Custom
decoding strategies can be implemented by subclassing :class:`BaseDecoder`.

.. autosummary::
    :toctree: ../generated/api

    BaseDecoder
    HysteresisDecoder

Interval
~~~~~~~~

:class:`Interval` is a simple data structure representing a detected or annotated call
with onset and offset times in seconds.

.. autosummary::
    :toctree: ../generated/api

    Interval
