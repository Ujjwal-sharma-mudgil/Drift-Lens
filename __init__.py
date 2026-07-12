"""DriftLens: statistical data-drift and anomaly detection for tabular data."""
from core import compare_datasets, compare_column, DriftReport, ColumnDriftResult, DriftStatus, ColumnType
from anomaly import detect, detect_dataframe, AnomalyResult
__version__ = "0.1.0"
__all__ = [
    "compare_datasets",
    "compare_column",
    "DriftReport",
    "ColumnDriftResult",
    "DriftStatus",
    "ColumnType",
    "detect",
    "detect_dataframe",
    "AnomalyResult",
    "__version__",
]
