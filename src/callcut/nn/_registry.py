"""Model registry for call detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from callcut.utils._checks import check_type, check_value

if TYPE_CHECKING:
    from callcut.nn._base import BaseDetector

_MODEL_REGISTRY: dict[str, type[BaseDetector]] = {}


def register_model(name: str):
    """Register a model class in the registry.

    This decorator adds a model class to the global registry, making it available
    via :func:`get_model` and :func:`~callcut.nn.list_models`.

    Parameters
    ----------
    name : str
        Name to register the model under. This name is used by
        :func:`~callcut.nn.get_model` to instantiate the model.

    Returns
    -------
    decorator : callable
        The decorator function.

    Examples
    --------
    Register a custom model:

    >>> from callcut.nn import BaseDetector, register_model
    >>> @register_model("my_custom_model")
    ... class MyCustomModel(BaseDetector):
    ...     # implementation
    ...     pass
    """
    check_type(name, (str,), "name")
    if len(name) == 0:
        raise ValueError("Model name cannot be empty.")

    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. Use a different name or "
                "unregister the existing model first."
            )
        _MODEL_REGISTRY[name] = cls

        # store the registered name on the class for reference
        cls._registered_name = name
        return cls

    return decorator


def get_model(name: str, **kwargs) -> BaseDetector:
    """Get a model instance by name.

    Parameters
    ----------
    name : str
        Name of the registered model.
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

    Examples
    --------
    >>> model = get_model("tiny_cnn", n_bands=8)
    >>> model = get_model("tiny_cnn", n_bands=8, base=64)

    See Also
    --------
    list_models : List all available model names.
    register_model : Register a new model.
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

    Examples
    --------
    >>> list_models()
    ['tiny_cnn']

    See Also
    --------
    get_model : Get a model instance by name.
    """
    return sorted(_MODEL_REGISTRY.keys())


def unregister_model(name: str) -> None:
    """Remove a model from the registry.

    Parameters
    ----------
    name : str
        Name of the model to unregister.

    Examples
    --------
    >>> unregister_model("my_custom_model")

    Raises
    ------
    ValueError
        If the model name is not found in the registry.
    """
    check_type(name, (str,), "name")
    check_value(name, _MODEL_REGISTRY, "name")
    del _MODEL_REGISTRY[name]
