Pipeline
--------

.. currentmodule:: callcut.pipeline

``callcut.pipeline`` provides end-to-end evaluation, prediction, and serialization
for trained call detection pipelines.

Evaluation and prediction
~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~callcut.pipeline.evaluate_recordings` runs a full evaluation pipeline on
annotated recordings: inference, decoding, interval matching, and metric computation.
Use it with :attr:`~callcut.training.CallDataModule.test_recordings` to evaluate on
held-out data after training.

:func:`~callcut.pipeline.predict_recordings` runs inference on new audio files without
ground truth.

.. autosummary::
    :toctree: ../generated/api

    evaluate_recordings
    predict_recordings

Serialization
~~~~~~~~~~~~~

:func:`~callcut.pipeline.save_pipeline` saves a complete pipeline (model, extractor,
decoder) to a single file. :func:`~callcut.pipeline.load_pipeline` reconstructs all
components from the saved file.

.. autosummary::
    :toctree: ../generated/api

    save_pipeline
    load_pipeline

Result types
~~~~~~~~~~~~

.. autosummary::
    :toctree: ../generated/api

    EvaluationReport
    RecordingEvaluation
    RecordingPrediction
