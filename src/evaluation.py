"""Model Evaluation Module.

This module provides utilities for evaluating machine learning
models including metrics calculation and cross-validation.
"""

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from typing import Any


class ModelEvaluator:
    """Handles model evaluation and metrics calculation."""

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Dictionary of metrics.
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1": f1_score(y_true, y_pred, average="weighted")
        }

    @staticmethod
    def evaluate_regression(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Dictionary of metrics.
        """
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate classification report.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Report string.
        """
        return classification_report(y_true, y_pred)

    @staticmethod
    def cross_validate(model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5) -> dict[str, Any]:
        """Perform cross-validation.

        Args:
            model: Scikit-learn model.
            X: Features.
            y: Labels.
            cv: Number of folds.

        Returns:
            Cross-validation results.
        """
        scores = cross_val_score(model, X, y, cv=cv)

        return {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "std": float(scores.std())
        }


def create_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "classification"
) -> dict[str, Any]:
    """Create comprehensive evaluation report.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        task_type: Task type (classification, regression).

    Returns:
        Evaluation report dictionary.
    """
    evaluator = ModelEvaluator()

    if task_type == "classification":
        metrics = evaluator.evaluate_classification(y_true, y_pred)
        report = evaluator.classification_report(y_true, y_pred)
    else:
        metrics = evaluator.evaluate_regression(y_true, y_pred)
        report = ""

    return {
        "task_type": task_type,
        "metrics": metrics,
        "classification_report": report,
        "sample_size": len(y_true)
    }
