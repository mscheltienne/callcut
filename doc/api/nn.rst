Model
-----

.. currentmodule:: callcut.nn

``callcut.nn`` provides a set of neural network models and utilities for building and
training neural networks. It includes a base detector class, a simple CNN model, and a
registry for managing custom models.

The :class:`~callcut.nn.BaseDetector` class implements the I/O roundtrip to save and
load registered models.

Built-in models
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated/api
    :template: autosummary/class_no_inherited_members.rst

    BaseDetector
    TinySegCNN

The following utility allow an I/O roundtrip of the models:

.. autosummary::
    :toctree: ../generated/api

    load_model
    save_model

Registry
~~~~~~~~

.. autosummary::
    :toctree: ../generated/api

    register_model
    unregister_model
    list_models
    get_model
