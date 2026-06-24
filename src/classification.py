"""Classification Models Module.

This module provides various classification algorithms
including logistic regression, decision trees, and SVMs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Any


class ModelClassifier:
    """Handles classification model training and prediction."""

    def __init__(self, model_type: str = "random_forest"):
        """Initialize the ModelClassifier.

        Args:
            model_type: Type of classifier (logistic, decision_tree, random_forest, svm, knn).
        """
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        """Create the classifier instance.

        Returns:
            Scikit-learn classifier.
        """
        if self.model_type == "logistic":
            return LogisticRegression(max_iter=1000)
        elif self.model_type == "decision_tree":
            return DecisionTreeClassifier(max_depth=10)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10)
        elif self.model_type == "svm":
            return SVC(kernel='rbf', probability=True)
        elif self.model_type == "knn":
            return KNeighborsClassifier(n_neighbors=5)
        return RandomForestClassifier(n_estimators=100)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the classifier.

        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X_test: Test features.

        Returns:
            Predicted labels.
        """
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Get prediction probabilities.

        Args:
            X_test: Test features.

        Returns:
            Probability predictions.
        """
        return self.model.predict_proba(X_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate accuracy score.

        Args:
            X_test: Test features.
            y_test: True labels.

        Returns:
            Accuracy score.
        """
        return self.model.score(X_test, y_test)

    def get_feature_importance(self) -> np.ndarray | None:
        """Get feature importance scores.

        Returns:
            Feature importance array or None.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        return None


def train_and_evaluate(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict[str, Any]:
    """Train and evaluate a classifier.

    Args:
        model_type: Type of classifier.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with training results.
    """
    clf = ModelClassifier(model_type=model_type)
    clf.train(X_train, y_train)

    predictions = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)

    return {
        "model_type": model_type,
        "accuracy": accuracy,
        "predictions": predictions,
        "feature_importance": clf.get_feature_importance()
    }
