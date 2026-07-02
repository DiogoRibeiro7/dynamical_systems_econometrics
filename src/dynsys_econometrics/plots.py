"""Plotting helpers for article figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_series_with_threshold(series: pd.Series, threshold: float, output_path: str | Path) -> None:
    """Save a time-series plot with a rare-event threshold line."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")

    fig, ax = plt.subplots(figsize=(10, 4))
    series.plot(ax=ax)
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_title("Rare-event threshold")
    ax.set_xlabel("date")
    ax.set_ylabel(series.name or "value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
