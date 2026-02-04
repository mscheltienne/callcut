"""Evaluation tools for call detection: decoding, matching, and metrics."""

from callcut.evaluation._boundary import compute_boundary_accuracy
from callcut.evaluation._decoding import BaseDecoder, HysteresisDecoder
from callcut.evaluation._event import compute_event_metrics
from callcut.evaluation._frame import compute_frame_metrics
from callcut.evaluation._matching import BaseIntervalMatcher, IoUMatcher
from callcut.evaluation._types import (
    BoundaryAccuracy,
    EventMetrics,
    FrameMetrics,
    Interval,
    Match,
)
