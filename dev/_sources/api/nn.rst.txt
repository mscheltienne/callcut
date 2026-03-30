Model
-----

.. currentmodule:: callcut.nn

``callcut.nn`` provides neural network models for call detection.

The :class:`~callcut.nn.BaseDetector` abstract class defines the interface for all
models, including sliding-window inference via :meth:`~callcut.nn.BaseDetector.predict`.

Models
~~~~~~

.. autosummary::
    :toctree: ../generated/api
    :template: autosummary/class_no_inherited_members.rst

    BaseDetector
    TinySegCNN
