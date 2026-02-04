Metrics
-------

.. currentmodule:: callcut.metrics

``callcut.metrics`` provides tools for evaluating call detection performance at
different granularities: frame-level, event-level, and boundary accuracy.

Frame-level metrics
~~~~~~~~~~~~~~~~~~~

Frame-level metrics evaluate detection at the individual frame granularity. Each frame
is classified as either containing a call (positive) or not (negative), and compared
against ground truth labels. This produces a standard confusion matrix with true
positives (TP), false positives (FP), false negatives (FN), and true negatives (TN).

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

To compute event-level metrics, predicted intervals must first be matched to ground
truth intervals using a matching strategy (e.g., IoU-based matching).

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

Interval matching
~~~~~~~~~~~~~~~~~

To compute event-level metrics and boundary accuracy, predicted intervals must be
matched to ground truth intervals. The matching strategy determines which predictions
correspond to which ground truth events.

.. autosummary::
    :toctree: ../generated/api

    BaseIntervalMatcher
    IoUMatcher
    Match
