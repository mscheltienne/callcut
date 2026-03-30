Features
--------

.. currentmodule:: callcut.extractors

``callcut.extractors`` provides feature extraction from audio signals. Feature
extractors convert raw audio waveforms into frame-level representations suitable
for neural network input.

The :class:`~callcut.extractors.BaseExtractor` abstract class defines the interface
for all extractors, ensuring consistent handling of time-to-frame conversions.

Extractors
~~~~~~~~~~

.. autosummary::
    :toctree: ../generated/api

    BaseExtractor
    SNRExtractor
