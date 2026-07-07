# ML Models Hub

A scikit-learn wrapper that exposes classification, regression, clustering, and evaluation through a uniform interface. Swap models by name, compare results, find what works for your data.

## Usage

```python
from src.classification import ModelClassifier
from src.evaluation import ModelEvaluator
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
clf = ModelClassifier(model_type="random_forest")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(ModelEvaluator.evaluate_classification(y_test, preds))
# -> {"accuracy": 0.96, "precision": 0.96, "recall": 0.96, "f1": 0.96}
```

## Supported Models

| Task | Algorithms |
|---|---|
| Classification | Logistic Regression, Decision Tree, Random Forest, SVM, KNN |
| Regression | Linear, Ridge, Lasso, ElasticNet, Gradient Boosting, SVR |
| Clustering | K-Means, Hierarchical, DBSCAN, Gaussian Mixture |

## License

MIT
