.. include:: ./links.inc

Usage
=====

``callcut`` detects animal calls in audio recordings. Each audio file needs a
companion annotation CSV (e.g. ``recording_annotations.csv``) with
``start_seconds`` and ``stop_seconds`` columns (values in milliseconds).

Training
--------

.. code-block:: python

    from pathlib import Path

    import lightning as L

    from callcut.evaluation import HysteresisDecoder, IoUMatcher
    from callcut.extractors import SNRExtractor
    from callcut.nn import TinySegCNN
    from callcut.pipeline import evaluate_recordings, save_pipeline
    from callcut.training import (
        BCEWithLogitsLoss,
        CallDataModule,
        CallDetectorModule,
        LoggingCallback,
        SaveBestModelCallback,
    )

    L.seed_everything(42)

    wav_files = sorted(Path("data/").glob("*.wav"))

    # Feature extraction: multi-band SNR
    extractor = SNRExtractor(sample_rate=32_000)

    # Data module: handles train/val/test splitting at the recording level
    dm = CallDataModule(recordings=wav_files, extractor=extractor, num_workers=0)
    dm.setup("fit")

    # Model: lightweight 1D CNN
    window_frames = extractor.seconds_to_frames(2.0)
    model = TinySegCNN(n_bands=extractor.n_features, window_frames=window_frames)
    module = CallDetectorModule(model, loss=BCEWithLogitsLoss())

    # Train
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="cpu",
        devices=1,
        callbacks=[LoggingCallback(), SaveBestModelCallback("best_weights.pt")],
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=dm)

    # Save the full pipeline (model + extractor + decoder config)
    decoder = HysteresisDecoder()
    save_pipeline(model, extractor, decoder, "pipeline.pt")

Evaluation
----------

After training, evaluate on the held-out test split using event-level, frame-level,
and boundary metrics:

.. code-block:: python

    from callcut.evaluation import HysteresisDecoder, IoUMatcher
    from callcut.pipeline import evaluate_recordings, load_pipeline

    model, extractor, decoder = load_pipeline("pipeline.pt")
    # dm.test_recordings is available after dm.setup("fit")
    report = evaluate_recordings(
        model, extractor, dm.test_recordings, decoder, IoUMatcher()
    )
    print(report)

The :class:`~callcut.pipeline.EvaluationReport` contains aggregated
:class:`~callcut.evaluation.EventMetrics`,
:class:`~callcut.evaluation.FrameMetrics`, and
:class:`~callcut.evaluation.BoundaryAccuracy` across all recordings, as well as
per-recording results.

Inference on new recordings
---------------------------

To run a trained pipeline on new audio files (no annotations needed):

.. code-block:: python

    from pathlib import Path

    from callcut.pipeline import load_pipeline, predict_recordings

    model, extractor, decoder = load_pipeline("pipeline.pt")
    audio_files = sorted(Path("new_data/").glob("*.wav"))
    predictions = predict_recordings(model, extractor, audio_files, decoder)

    for pred in predictions:
        print(f"{pred.audio_path.name}: {len(pred.intervals)} calls")
        for interval in pred.intervals:
            print(f"  {interval.onset:.3f}s - {interval.offset:.3f}s")

Each :class:`~callcut.pipeline.RecordingPrediction` contains the detected call
:class:`~callcut.evaluation.Interval` objects with ``onset`` and ``offset``
times in seconds.
