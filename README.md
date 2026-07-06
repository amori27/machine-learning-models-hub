# Machine Learning Models Hub

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/amori27/machine-learning-models-hub/actions/workflows/ci.yml/badge.svg)](https://github.com/amori27/machine-learning-models-hub/actions/workflows/ci.yml)

A unified scikit-learn wrapper for the most common ML tasks: classification, regression, clustering, and evaluation. Pick a model by name, get a consistent interface, swap and benchmark freely.

---

## Modules

| Module | Algorithms |
|---|---|
| `classification.ModelClassifier` | Logistic Regression, Decision Tree, Random Forest, SVM, KNN |
| `regression.ModelRegressor` | Linear, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, SVR |
| `clustering.ModelClusterer` | K-Means, Hierarchical, DBSCAN, Gaussian Mixture |
| `evaluation.ModelEvaluator` | Accuracy, precision/recall/F1, RMSE/MAE/R², cross-validation |

---

## Install

```bash
pip install -r requirements.txt
```

---

## Usage

### Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.classification import ModelClassifier
from src.evaluation import ModelEvaluator

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = ModelClassifier(model_type="random_forest")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print(ModelEvaluator.evaluate_classification(y_test, preds))
# -> {"accuracy": 0.96, "precision": 0.96, "recall": 0.96, "f1": 0.96}
```

### Regression

```python
from src.regression import ModelRegressor

reg = ModelRegressor(model_type="gradient_boosting")
reg.fit(X_train, y_train)
preds = reg.predict(X_test)

print(ModelEvaluator.evaluate_regression(y_test, preds))
```

### Clustering

```python
from src.clustering import ModelClusterer

clusterer = ModelClusterer(algorithm="kmeans", n_clusters=3)
labels = clusterer.fit_predict(X)
```

---

## Project Structure

```
machine-learning-models-hub/
├── src/
│   ├── classification.py
│   ├── regression.py
│   ├── clustering.py
│   └── evaluation.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).
