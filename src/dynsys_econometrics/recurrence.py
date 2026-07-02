"""Convenience wrappers for recurrence diagnostics."""

from __future__ import annotations

from dynsys_econometrics.return_times import (
    compare_return_time_distributions,
    inter_event_durations,
    recurrence_rate,
    return_times,
    survival_curve,
)

__all__ = [
    "compare_return_time_distributions",
    "inter_event_durations",
    "recurrence_rate",
    "return_times",
    "survival_curve",
]
