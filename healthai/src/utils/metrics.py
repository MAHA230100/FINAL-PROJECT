from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, silhouette_score, calinski_harabasz_score


def classification_metrics(y_true, y_pred, y_proba: np.ndarray | None = None) -> Dict[str, float]:
	metrics = {
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"f1": float(f1_score(y_true, y_pred, average="weighted")),
	}
	if y_proba is not None and len(np.unique(y_true)) == 2:
		metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
	return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	mae = float(mean_absolute_error(y_true, y_pred))
	r2 = float(r2_score(y_true, y_pred))
	return {"rmse": rmse, "mae": mae, "r2": r2}


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
	return {
		"silhouette": float(silhouette_score(X, labels)),
		"calinski_harabasz": float(calinski_harabasz_score(X, labels)),
	} 