"""
Muraqib — Model Evaluation Script
Runs a full evaluation suite on the RandomForest delay-prediction model.
Usage: python evaluate_model.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  # kept for cross_val_score estimator type
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from muraqib.data_loader import load_data, get_feature_columns
from muraqib.model import _train  # reuse exact same training logic as the app

# ─── Helpers ────────────────────────────────────────────────────────────────────

def separator(title=""):
    line = "-" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


# ─── Load Data ──────────────────────────────────────────────────────────────────

separator("1. Loading Data")
df = load_data()
features = get_feature_columns()

X = df[features]
y = df["is_delayed"]

print(f"  Total samples       : {len(df)}")
print(f"  Features used       : {features}")
print(f"  Class distribution  :")
counts = y.value_counts()
for label, count in counts.items():
    tag = "Delayed" if label == 1 else "On-Time"
    print(f"    [{label}] {tag:10s} -> {count} samples ({pct(count / len(y))})")


# ─── Train / Test Split ─────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

separator("2. Train / Test Split")
print(f"  Training set size   : {len(X_train)} ({pct(len(X_train)/len(X))})")
print(f"  Test set size       : {len(X_test)}  ({pct(len(X_test)/len(X))})")


# --- Train Model (always mirrors model.py hyperparams) -------------------------

# _train() builds the model with the exact same hyperparams as the live app.
# Any change you make to model.py is automatically picked up here.
clf, _train_reported_acc = _train(df)

print(f"  Model params        : n_estimators={clf.n_estimators}, "
      f"max_depth={clf.max_depth}, "
      f"min_samples_split={clf.min_samples_split}, "
      f"min_samples_leaf={clf.min_samples_leaf}")

# _train() uses its own 80/20 split internally.
# We re-predict on our stratified X_test for a fair comparison.
y_pred       = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]


# ─── Core Metrics ───────────────────────────────────────────────────────────────

separator("3. Core Metrics (Test Set)")
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print(f"  Accuracy            : {pct(accuracy)}")
print(f"  Precision           : {pct(precision)}")
print(f"  Recall              : {pct(recall)}")
print(f"  F1 Score            : {pct(f1)}")
print(f"  ROC-AUC             : {pct(roc_auc)}")


# ─── Confusion Matrix ───────────────────────────────────────────────────────────

separator("4. Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"""
                   Predicted
                On-Time  Delayed
  Actual On-Time  {tn:4d}    {fp:4d}    (TN / FP)
  Actual Delayed  {fn:4d}    {tp:4d}    (FN / TP)
""")
print(f"  True  Positives (TP) — Correctly flagged as delayed   : {tp}")
print(f"  True  Negatives (TN) — Correctly flagged as on-time   : {tn}")
print(f"  False Positives (FP) — Wrongly flagged as delayed      : {fp}")
print(f"  False Negatives (FN) — Missed delays                   : {fn}")


# ─── Full Classification Report ─────────────────────────────────────────────────

separator("5. Full Classification Report")
print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))


# ─── Cross-Validation ───────────────────────────────────────────────────────────

separator("6. Stratified 5-Fold Cross-Validation")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for metric_name, scoring in [
    ("Accuracy ", "accuracy"),
    ("Precision", "precision"),
    ("Recall   ", "recall"),
    ("F1 Score ", "f1"),
    ("ROC-AUC  ", "roc_auc"),
]:
    scores = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
    print(f"  {metric_name} -> mean: {pct(scores.mean())}  std: +/- {pct(scores.std())}  folds: {[f'{s*100:.1f}%' for s in scores]}")


# ─── Feature Importances ────────────────────────────────────────────────────────

separator("7. Feature Importances")
importances = clf.feature_importances_
fi_sorted = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
for feat, imp in fi_sorted:
    bar = "#" * int(imp * 50)
    print(f"  {feat:<30s} {imp:.4f}  {bar}")


# ─── Overfitting Check ──────────────────────────────────────────────────────────

separator("8. Overfitting Check (Train vs Test Accuracy)")
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc  = accuracy_score(y_test,  y_pred)
gap       = train_acc - test_acc
print(f"  Train Accuracy      : {pct(train_acc)}")
print(f"  Test  Accuracy      : {pct(test_acc)}")
print(f"  Gap (Train - Test)  : {pct(gap)}")
if gap < 0.05:
    print("  ✅ No significant overfitting detected.")
elif gap < 0.15:
    print("  ⚠️  Mild overfitting. Consider tuning max_depth or min_samples_leaf.")
else:
    print("  ❌ Significant overfitting. Model may be memorising training data.")


separator()
print("  Evaluation complete.")
separator()
