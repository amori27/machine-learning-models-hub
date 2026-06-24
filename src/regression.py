"""Regression Models Module.

This module provides various regression algorithms
including linear regression, ridge, lasso, and tree-based methods.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from typing import Any


class ModelRegressor:
    """Handles regression model training and prediction."""

    def __init__(self, model_type: str = "linear"):
        """Initialize the ModelRegressor.

        Args:
            model_type: Type of regressor (linear, ridge, lasso, elasticnet, decision_tree, random_forest, gradient_boosting, svr).
        """
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        """Create the regressor instance.

        Returns:
            Scikit-learn regressor.
        """
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "lasso":
            return Lasso(alpha=1.0)
        elif self.model_type == "elasticnet":
            return ElasticNet(alpha=1.0, l1_ratio=0.5)
        elif self.model_type == "decision_tree":
            return DecisionTreeRegressor(max_depth=10)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators=100, max_depth=10)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators=100, max_depth=5)
        elif self.model_type == "svr":
            return SVR(kernel='rbf')
        return LinearRegression()

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the regressor.

        Args:
            X_train: Training features.
            y_train: Training target values.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X_test: Test features.

        Returns:
            Predicted values.
        """
        return self.model.predict(X_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate R² score.

        Args:
            X_test: Test features.
            y_test: True target values.

        Returns:
            R² score.
        """
        return self.model.score(X_test, y_test)

    def get_coefficients(self) -> np.ndarray | None:
        """Get model coefficients.

        Returns:
            Coefficients array or None.
        """
        if hasattr(self.model, 'coef_'):
            return self.model.coef_
        elif hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


def train_and_evaluate_regression(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict[str, Any]:
    """Train and evaluate a regressor.

    Args:
        model_type: Type of regressor.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary with training results.
    """
    reg = ModelRegressor(model_type=model_type)
    reg.train(X_train, y_train)

    predictions = reg.predict(X_test)
    r2_score = reg.score(X_test, y_test)

    return {
        "model_type": model_type,
        "r2_score": r2_score,
        "predictions": predictions,
        "coefficients": reg.get_coefficients()
    }
