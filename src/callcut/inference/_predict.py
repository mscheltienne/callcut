"""Inference functions for running models on full recordings."""

from __future__ import annotations

import torch
from torch import Tensor

from callcut.nn import BaseDetector
from callcut.utils._checks import check_type, ensure_int


def predict_recording(
    model: BaseDetector,
    features: Tensor,
    *,
    window_frames: int,
    window_hop_frames: int,
) -> Tensor:
    """Run inference on a full recording using sliding windows.

    The model is applied to overlapping windows across the recording. Where
    windows overlap, predictions are averaged to produce smoother, more robust
    per-frame probability estimates.

    Parameters
    ----------
    model : BaseDetector
        Trained call detection model.
    features : Tensor
        Input features of shape ``(n_bands, n_frames)``. Should be on the same
        device as the model.
    window_frames : int
        Window size in frames. Should match the window size used during training.
    window_hop_frames : int
        Hop between consecutive windows in frames. Smaller values produce more
        overlap and smoother predictions but increase computation time.

    Returns
    -------
    probabilities : Tensor
        Per-frame call probabilities of shape ``(n_frames,)``. Values are in
        ``[0, 1]``, where higher values indicate higher confidence that a call
        is present.

    Notes
    -----
    The inference process:

    1. Slide a window of size ``window_frames`` across the recording with step
       ``window_hop_frames``.
    2. For each window, run the model to get logits, then apply sigmoid to get
       probabilities.
    3. Accumulate predictions for each frame. Frames covered by multiple windows
       receive multiple predictions.
    4. Average the accumulated predictions to get final per-frame probabilities.

    For frames near the end of the recording that don't fit a full window, the
    window is padded using edge values.

    Examples
    --------
    >>> from callcut.nn import load_model
    >>> from callcut.features import SNRExtractor
    >>> from callcut.io import load_audio
    >>> from callcut.inference import predict_recording
    >>>
    >>> model = load_model("detector.pt", device="cpu")
    >>> extractor = SNRExtractor(sample_rate=32000, hop_ms=8.0, n_bands=8)
    >>>
    >>> waveform, sr = load_audio("recording.wav", sample_rate=32000)
    >>> features, times = extractor(waveform)
    >>>
    >>> probs = predict_recording(
    ...     model,
    ...     features,
    ...     window_frames=250,  # 2 seconds at 8ms hop
    ...     window_hop_frames=62,  # 0.5 seconds
    ... )
    >>> probs.shape
    torch.Size([1234])
    """
    check_type(model, (BaseDetector,), "model")
    check_type(features, (Tensor,), "features")
    window_frames = ensure_int(window_frames, "window_frames")
    window_hop_frames = ensure_int(window_hop_frames, "window_hop_frames")

    if window_frames <= 0:
        raise ValueError(
            f"Argument 'window_frames' must be positive, got {window_frames}."
        )
    if window_hop_frames <= 0:
        raise ValueError(
            f"Argument 'window_hop_frames' must be positive, got {window_hop_frames}."
        )
    if window_hop_frames > window_frames:
        raise ValueError(
            f"Argument 'window_hop_frames' ({window_hop_frames}) cannot be greater "
            f"than 'window_frames' ({window_frames})."
        )

    if features.dim() != 2:
        raise ValueError(
            f"features must be 2D (n_bands, n_frames), got {features.dim()}D."
        )

    device = features.device
    dtype = features.dtype
    n_bands, n_frames = features.shape

    if n_bands != model.n_bands:
        raise ValueError(
            f"Feature bands ({n_bands}) do not match model input bands "
            f"({model.n_bands})."
        )

    # Initialize accumulators
    prob_sum = torch.zeros(n_frames, dtype=dtype, device=device)
    prob_count = torch.zeros(n_frames, dtype=dtype, device=device)

    # Compute window start positions
    # Generate starts so that windows cover the recording
    if n_frames <= window_frames:
        starts = [0]
    else:
        starts = list(range(0, n_frames - window_frames + 1, window_hop_frames))
        # Ensure we have at least one start
        if not starts:
            starts = [0]

    # Run inference
    model.eval()
    with torch.no_grad():
        for start in starts:
            end = start + window_frames

            # Extract window
            if end <= n_frames:
                window = features[:, start:end]
            else:
                # Pad with edge values if window extends beyond recording
                available = features[:, start:n_frames]
                pad_size = end - n_frames
                # Use edge padding (repeat last frame)
                padding = features[:, -1:].expand(-1, pad_size)
                window = torch.cat([available, padding], dim=1)

            # Run model: (n_bands, window_frames) -> add batch dim -> model -> remove
            logits = model(window.unsqueeze(0)).squeeze(0)  # (window_frames,)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Accumulate (only for valid frames, not padding)
            valid_end = min(n_frames, end)
            valid_len = valid_end - start
            prob_sum[start:valid_end] += probs[:valid_len]
            prob_count[start:valid_end] += 1.0

    # Average predictions
    # Avoid division by zero (shouldn't happen if starts is computed correctly)
    prob_count = torch.clamp(prob_count, min=1e-12)
    probabilities = prob_sum / prob_count

    return probabilities
