## Data Split Decision

Initial approach: global temporal split across concatenated 24-series dataset.
Problem: split at timestamp boundaries left 1.1% anomaly rate in validation,
making threshold calibration unreliable.

Resolution: per-series temporal split (define_temporal_split_per_series).
Each series split independently at 70/15/15 by timestamp.

Outcome:
  Train: 7.4%  Val: 9.7%  Test: 18.6%

Test imbalance is a known property of NAB realKnownCause series (rogue_agent_key_hold,
ambient_temperature_system_failure) where anomaly events cluster in the latter
portion of the monitoring period. AUC-ROC is used as the primary evaluation
metric rather than F1 because it is threshold-independent and robust to
class imbalance.