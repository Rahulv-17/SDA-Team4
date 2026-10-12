"""
Model factories for streaming classifiers.
Supports:
- Logistic Regression with Standard Scaler (primary, rehearsal-friendly)
- Hoeffding Adaptive Tree Classifier (HAT)
"""
from river import preprocessing, linear_model, tree


def create_base_model(model_type: str = "logistic", seed: int = 42):
    """Instantiate a fresh base model."""
    m = model_type.lower()
    if m in ("logistic", "lr"):
        return preprocessing.StandardScaler() | linear_model.LogisticRegression()
    elif m in ("hat", "tree"):
        return tree.HoeffdingAdaptiveTreeClassifier(seed=seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Supported: 'logistic', 'hat'")
