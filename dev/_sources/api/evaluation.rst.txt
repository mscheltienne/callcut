Evaluation
----------

.. currentmodule:: callcut.evaluation

``callcut.evaluation`` provides tools for evaluating call detection performance:
decoding predictions into intervals, matching intervals, and computing metrics.

.. note::

    For running inference on full recordings, use the
    :meth:`~callcut.nn.BaseDetector.predict` method on the model directly.

Decoding
~~~~~~~~

Decoders convert frame-level probabilities (values in ``[0, 1]`` for each time frame)
into a list of discrete call intervals with onset and offset times.

:class:`~callcut.evaluation.HysteresisDecoder` uses hysteresis thresholding with
separate enter/exit thresholds to avoid rapid on/off switching when probabilities hover
near a single threshold. It also merges nearby intervals and filters out short
detections. Custom decoding strategies can be implemented by subclassing
:class:`~callcut.evaluation.BaseDecoder`.

.. autosummary::
    :toctree: ../generated/api

    BaseDecoder
    HysteresisDecoder

Interval matching
~~~~~~~~~~~~~~~~~

To compute event-level metrics and boundary accuracy, predicted intervals must be
matched to ground truth intervals. The matching strategy determines which predictions
correspond to which ground truth events.

:class:`~callcut.evaluation.IoUMatcher` uses greedy Intersection-over-Union (IoU)
matching: it pairs predictions with ground truth based on their temporal overlap,
prioritizing high-overlap matches. Custom matching strategies can be implemented by
subclassing :class:`~callcut.evaluation.BaseIntervalMatcher`.

.. autosummary::
    :toctree: ../generated/api

    BaseIntervalMatcher
    IoUMatcher

Frame-level metrics
~~~~~~~~~~~~~~~~~~~

Frame-level metrics evaluate detection at the individual frame granularity. Each frame
is classified as either containing a call (positive) or not (negative), and compared
against ground truth labels. This produces a standard confusion matrix with true
positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).

Frame-level metrics are useful for quick sanity checks during training, but can be
misleading for event detection. A model might achieve high frame-level F1 by correctly
predicting the middle of calls while missing boundaries, or by predicting many short
false alarms that happen to overlap with calls.

.. autosummary::
    :toctree: ../generated/api

    compute_frame_metrics
    FrameMetrics

Event-level metrics
~~~~~~~~~~~~~~~~~~~

Event-level metrics evaluate detection at the call/event granularity. Each ground truth
call is either matched to a prediction (true positive) or missed (false negative), and
each prediction is either matched (true positive) or a false alarm (false positive).

Event-level metrics better reflect real-world performance: they answer questions like
"How many calls did we detect?" and "How many false alarms did we produce?" rather than
"What fraction of frames were correct?".

.. autosummary::
    :toctree: ../generated/api

    compute_event_metrics
    EventMetrics

Boundary metrics
~~~~~~~~~~~~~~~~

Boundary accuracy measures how precisely the predicted call boundaries (onset and offset
times) align with the ground truth boundaries. This is computed only for matched events
(true positives).

Errors are computed as ``predicted - ground_truth``, so:

- **Positive onset error**: The prediction started too late
- **Negative onset error**: The prediction started too early
- **Positive offset error**: The prediction ended too late
- **Negative offset error**: The prediction ended too early

Boundary accuracy complements event-level metrics: a model might detect all calls
(perfect recall) but consistently start predictions 100ms late, which boundary accuracy
would reveal.

.. autosummary::
    :toctree: ../generated/api

    compute_boundary_accuracy
    BoundaryAccuracy

Types
~~~~~

Core data types used throughout the evaluation module.

.. autosummary::
    :toctree: ../generated/api

    Interval
    Match
