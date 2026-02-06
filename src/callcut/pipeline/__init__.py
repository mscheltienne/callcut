"""Pipeline module for end-to-end evaluation, prediction, and serialization."""

from callcut.pipeline._evaluate import evaluate_recordings
from callcut.pipeline._predict import predict_recordings
from callcut.pipeline._save_load import load_pipeline, save_pipeline
from callcut.pipeline._types import (
    EvaluationReport,
    RecordingEvaluation,
    RecordingPrediction,
)
