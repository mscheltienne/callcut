"""Model registry for call detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from callcut.utils._checks import check_type, check_value

if TYPE_CHECKING:
    from callcut.nn._base import BaseDetector

_MODEL_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_model(cls: type[BaseDetector]) -> type[BaseDetector]:
    """Register a model class in the registry.

    This decorator adds a model class to the global registry using the class name,
    making it available via :func:`get_model` and :func:`list_models`.

    Parameters
    ----------
    cls : type
        The model class to register.

    Returns
    -------
    cls : type
        The same class, unmodified.

    Examples
    --------
    Register a custom model:

    >>> from callcut.nn import BaseDetector, register_model
    >>> @register_model
    ... class MyCustomModel(BaseDetector):
    ...     # implementation
    ...     pass

    The model is then available via:

    >>> model = get_model("MyCustomModel", n_bands=8)
    """
    name = cls.__name__
    if name in _MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered. Use a different class name or "
            "unregister the existing model first."
        )
    _MODEL_REGISTRY[name] = cls
    return cls


def get_model(name: str, **kwargs) -> BaseDetector:
    """Get a model instance by name.

    Parameters
    ----------
    name : str
        Name of the registered model (the class name).
    **kwargs
        Keyword arguments passed to the model constructor.

    Returns
    -------
    model : BaseDetector
        An instance of the requested model.

    Raises
    ------
    ValueError
        If the model name is not found in the registry.

    See Also
    --------
    list_models : List all available model names.
    register_model : Register a new model.

    Examples
    --------
    >>> model = get_model("TinySegCNN", n_bands=8)
    >>> model = get_model("TinySegCNN", n_bands=8, base=64)
    """
    check_type(name, (str,), "name")
    check_value(name, _MODEL_REGISTRY, "name")
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """List all registered model names.

    Returns
    -------
    models : list of str
        Names of all registered models, sorted alphabetically.

    See Also
    --------
    get_model : Get a model instance by name.

    Examples
    --------
    >>> list_models()
    ['TinySegCNN']
    """
    return sorted(_MODEL_REGISTRY.keys())


def unregister_model(name: str) -> None:
    """Remove a model from the registry.

    Parameters
    ----------
    name : str
        Name of the model to unregister.

    Raises
    ------
    ValueError
        If the model name is not found in the registry.

    Examples
    --------
    >>> unregister_model("MyCustomModel")
    """
    check_type(name, (str,), "name")
    check_value(name, _MODEL_REGISTRY, "name")
    del _MODEL_REGISTRY[name]
