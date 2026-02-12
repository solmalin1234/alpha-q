"""MLFlow experiment logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow


class ExperimentLogger:
    """Thin wrapper around MLFlow for experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: str = "mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()

    def log_params(self, params: dict[str, Any], prefix: str = "") -> None:
        """Log a (possibly nested) dict of parameters."""
        flat = self._flatten(params, prefix)
        # MLFlow has a 100-param batch limit
        items = list(flat.items())
        for i in range(0, len(items), 100):
            mlflow.log_params(dict(items[i : i + 100]))

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        mlflow.log_artifact(str(path))

    def end(self) -> None:
        mlflow.end_run()

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
        """Flatten a nested dict into dot-separated keys with string values."""
        items: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ExperimentLogger._flatten(v, key))
            else:
                items[key] = str(v)
        return items
