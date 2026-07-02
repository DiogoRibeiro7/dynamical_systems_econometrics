"""Tools for dynamical-systems inspired econometric rare-event analysis."""

from dynsys_econometrics.extremes import ExtremalIndexResult, estimate_runs_extremal_index
from dynsys_econometrics.extremes import (
    ExceedanceExtractionResult,
    ThresholdSensitivityResult,
    block_maxima,
    estimate_cluster_size_extremal_index,
    estimate_extremal_index,
    extract_threshold_exceedances,
    peaks_over_threshold,
    runs_declustering,
    runs_declustering_from_indicator,
    threshold_sensitivity_analysis,
)
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import (
    logistic_map,
    simulate_ar1,
    simulate_coupled_logistic_maps,
    simulate_garch11,
    simulate_iid_gaussian,
    simulate_pomeau_manneville,
    simulate_student_t,
)

__all__ = [
    "ExtremalIndexResult",
    "ExceedanceExtractionResult",
    "ThresholdSensitivityResult",
    "compute_return_times",
    "block_maxima",
    "estimate_runs_extremal_index",
    "estimate_cluster_size_extremal_index",
    "estimate_extremal_index",
    "extract_threshold_exceedances",
    "peaks_over_threshold",
    "runs_declustering",
    "runs_declustering_from_indicator",
    "threshold_sensitivity_analysis",
    "logistic_map",
    "simulate_ar1",
    "simulate_coupled_logistic_maps",
    "simulate_garch11",
    "simulate_iid_gaussian",
    "simulate_pomeau_manneville",
    "simulate_student_t",
]
