from preprocess import preprocess_pipeline
from train import train_models, select_best_model, save_model
from evaluate import (
    print_basic_metrics,
    plot_roc,
    find_threshold_with_constraints,
    feature_importance
)

# ========================
# 1. Preprocess
# ========================
X_train, X_test, y_train, y_test = preprocess_pipeline("data/real.csv")

# ========================
# 2. Train
# ========================
models = train_models(X_train, y_train)

# ========================
# 3. Select best
# ========================
best_model = select_best_model(models, X_test, y_test)

# ========================
# 4. Save
# ========================
save_model(best_model)

# ========================
# 5. Baseline evaluation
# ========================
print("\n🔹 Evaluation at default threshold (0.5)")
print_basic_metrics(best_model, X_test, y_test, threshold=0.5)

# ========================
# 6. ROC
# ========================
plot_roc(best_model, X_test, y_test)

# ========================
# 7. Precision optimization
# ========================
best_threshold = find_threshold_with_constraints(
    best_model, X_test, y_test, min_recall=0.6
)

# ========================
# 8. Final evaluation
# ========================
print(f"\n🔹 Evaluation at optimized threshold ({best_threshold:.2f})")
print_basic_metrics(best_model, X_test, y_test, threshold=best_threshold)

# ========================
# 9. Feature importance
# ========================
feature_importance(best_model, X_train.columns)