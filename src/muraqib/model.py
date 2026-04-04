import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .data_loader import get_feature_columns

_MODEL_CACHE = None


def _train(df: pd.DataFrame) -> RandomForestClassifier:
    features = get_feature_columns()
    X = df[features]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100,
                                min_samples_split=20, 
                                min_samples_leaf=15,
                                random_state=42, 
                                max_depth=8,
                                class_weight='balanced')
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    return clf, round(acc * 100, 1)


def get_model(df: pd.DataFrame):
    """Return cached (model, accuracy) tuple; train if not yet cached."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _train(df)
    return _MODEL_CACHE


def predict(
    model: RandomForestClassifier,
    complexity_enc: int,
    supply_delay_days: int,
    subcontractor_performance: float,
    weather_enc: int,
    labor_availability: float,
) -> dict:
    """Run prediction and return probability + feature importances."""
    X = np.array(
        [[complexity_enc, supply_delay_days, subcontractor_performance, weather_enc, labor_availability]]
    )
    proba = model.predict_proba(X)[0]
    delay_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
    is_delayed = delay_prob >= 0.5

    importances = model.feature_importances_
    features = get_feature_columns()
    importance_map = {f: float(v) for f, v in zip(features, importances)}

    return {
        "is_delayed": is_delayed,
        "delay_probability": round(delay_prob * 100, 1),
        "feature_importances": importance_map,
    }
