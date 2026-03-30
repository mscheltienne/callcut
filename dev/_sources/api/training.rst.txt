Training
--------

.. currentmodule:: callcut.training

``callcut.training`` provides PyTorch Lightning infrastructure for training call
detection models. It includes Lightning modules, data modules, loss functions,
and callbacks.

Lightning Modules
~~~~~~~~~~~~~~~~~

The :class:`~callcut.training.CallDetectorModule` wraps any
:class:`~callcut.nn.BaseDetector` model for training with PyTorch Lightning. It
handles the training loop, validation metrics, and optimizer configuration.

The :class:`~callcut.training.CallDataModule` handles data loading, train/val/test
splitting (balanced by window count), and DataLoader creation.

.. autosummary::
    :toctree: ../generated/api
    :template: autosummary/class_no_inherited_members.rst

    CallDetectorModule
    CallDataModule

Loss Functions
~~~~~~~~~~~~~~

Loss functions for training call detection models. All loss functions inherit from
:class:`~callcut.training.BaseLoss` and expect logits (raw model output before
sigmoid) and binary target labels.

.. autosummary::
    :toctree: ../generated/api
    :template: autosummary/class_no_inherited_members.rst

    BaseLoss
    BCEWithLogitsLoss
    FocalLoss
    DiceLoss
    TverskyLoss

Callbacks
~~~~~~~~~

Custom Lightning callbacks for training.

.. autosummary::
    :toctree: ../generated/api
    :template: autosummary/class_no_inherited_members.rst

    LoggingCallback
    MetricsHistoryCallback
    SaveBestModelCallback
