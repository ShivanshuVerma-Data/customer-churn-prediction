import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score
)


def print_basic_metrics(model, X_test, y_test, threshold=0.5):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    print(f"\n=== Classification Report (threshold={threshold:.2f}) ===")
    print(classification_report(y_test, y_pred, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Confusion Matrix ===")
    print(cm)


def plot_roc(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


def find_threshold_with_constraints(model, X_test, y_test, min_recall=0.5):
    y_probs = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.3, 0.95, 100)

    best_thresh = 0.5
    best_precision = 0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)

        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds)

        if recall >= min_recall and precision > best_precision:
            best_precision = precision
            best_thresh = t

    print(f"\nBest Threshold (precision-focused): {best_thresh:.2f}")
    print(f"Precision: {best_precision:.4f} (recall >= {min_recall})")

    return best_thresh


def feature_importance(model, X_columns):
    try:
        importances = model.feature_importances_
    except:
        print("\nFeature importance not available for this model.")
        return

    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [X_columns[i] for i in indices])
    plt.title("Top Feature Importances")
    plt.show()