"""Save and load complete pipelines (model + extractor + decoder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from callcut.evaluation import BaseDecoder, HysteresisDecoder
from callcut.extractors import BaseExtractor, SNRExtractor
from callcut.nn import BaseDetector, TinySegCNN
from callcut.utils._checks import check_type, check_value, ensure_device, ensure_path
from callcut.utils.logs import logger

if TYPE_CHECKING:
    from pathlib import Path

_EXTRACTOR_TYPES: dict[str, type[BaseExtractor]] = {
    "SNRExtractor": SNRExtractor,
}

_DECODER_TYPES: dict[str, type[BaseDecoder]] = {
    "HysteresisDecoder": HysteresisDecoder,
}

_MODEL_TYPES: dict[str, type[BaseDetector]] = {
    "TinySegCNN": TinySegCNN,
}


def save_pipeline(
    model: BaseDetector,
    extractor: BaseExtractor,
    decoder: BaseDecoder,
    fname: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Save a complete pipeline to a file.

    Saves the model architecture, trained weights, feature extractor configuration,
    and decoder configuration. The resulting file is self-contained:
    :func:`load_pipeline` can reconstruct all components without additional information.

    Parameters
    ----------
    model : BaseDetector
        The trained model.
    extractor : BaseExtractor
        Feature extractor used with this model.
    decoder : BaseDecoder
        Decoder for converting probabilities to intervals.
    fname : str | Path
        Path to save the pipeline. Conventionally use ``.pt`` extension.
    overwrite : bool
        If ``True``, overwrite the file if it exists.

    See Also
    --------
    load_pipeline : Load a pipeline from a file.

    Examples
    --------
    >>> from callcut.extractors import SNRExtractor
    >>> from callcut.nn import TinySegCNN
    >>> from callcut.evaluation import HysteresisDecoder
    >>> from callcut.pipeline import save_pipeline
    >>>
    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>> model = TinySegCNN(n_bands=8, window_frames=250)
    >>> decoder = HysteresisDecoder()
    >>> save_pipeline(model, extractor, decoder, "pipeline.pt")
    """
    check_type(model, (BaseDetector,), "model")
    check_type(extractor, (BaseExtractor,), "extractor")
    check_type(decoder, (BaseDecoder,), "decoder")
    fname = ensure_path(fname, must_exist=False)

    if not overwrite and fname.exists():
        raise FileExistsError(
            f"File {fname} already exists. Use overwrite=True to replace it."
        )

    logger.info("Saving pipeline to %s", fname)
    checkpoint: dict = {
        "model": {
            "class_name": model.__class__.__name__,
            "n_bands": model.n_bands,
            "window_frames": model.window_frames,
            "receptive_field": model.receptive_field,
            "config": model._save_config(),
            "state_dict": model.state_dict(),
        },
        "extractor": {
            "class_name": extractor.__class__.__name__,
            "config": extractor._save_config(),
        },
        "decoder": {
            "class_name": decoder.__class__.__name__,
            "config": decoder._save_config(),
        },
    }

    torch.save(checkpoint, fname)


def load_pipeline(
    fname: str | Path, *, device: str | torch.device | None = None
) -> tuple[BaseDetector, BaseExtractor, BaseDecoder]:
    """Load a complete pipeline from a file.

    Reconstructs the model (with trained weights), feature extractor, and
    decoder from a file saved with :func:`~callcut.pipeline.save_pipeline`.

    Parameters
    ----------
    fname : str | Path
        Path to the saved pipeline file.
    device : str | torch.device | None
        Device to load the model to (e.g., ``"cpu"``, ``"cuda:0"``, ``"mps"``).
        If ``None``, uses the default torch device.

    Returns
    -------
    model : BaseDetector
        The trained model with loaded weights.
    extractor : BaseExtractor
        Feature extractor matching the model's expected input.
    decoder : BaseDecoder
        Decoder for converting probabilities to intervals.

    See Also
    --------
    save_pipeline : Save a pipeline to a file.

    Examples
    --------
    >>> from callcut.pipeline import load_pipeline
    >>>
    >>> model, extractor, decoder = load_pipeline("pipeline.pt", device="cpu")
    """
    fname = ensure_path(fname, must_exist=True)
    device = ensure_device(device)
    logger.info("Loading pipeline from %s", fname)

    checkpoint = torch.load(fname, map_location=device, weights_only=False)

    # Reconstruct extractor
    ext_data = checkpoint["extractor"]
    ext_cls_name = ext_data["class_name"]
    check_value(ext_cls_name, _EXTRACTOR_TYPES, "extractor class_name")
    extractor = _EXTRACTOR_TYPES[ext_cls_name](**ext_data["config"])

    # Reconstruct model
    model_data = checkpoint["model"]
    model_cls_name = model_data["class_name"]
    check_value(model_cls_name, _MODEL_TYPES, "model class_name")
    model = _MODEL_TYPES[model_cls_name](
        n_bands=model_data["n_bands"],
        window_frames=model_data["window_frames"],
        **model_data.get("config", {}),
    )
    model.load_state_dict(model_data["state_dict"])
    model = model.to(device)
    logger.debug(
        "Loaded %s with n_bands=%d, window_frames=%d",
        model_data["class_name"],
        model_data["n_bands"],
        model_data["window_frames"],
    )

    # Reconstruct decoder
    dec_data = checkpoint["decoder"]
    dec_cls_name = dec_data["class_name"]
    check_value(dec_cls_name, _DECODER_TYPES, "decoder class_name")
    decoder = _DECODER_TYPES[dec_cls_name](**dec_data["config"])

    return model, extractor, decoder
