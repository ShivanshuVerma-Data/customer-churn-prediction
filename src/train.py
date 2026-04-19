import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def train_models(X_train, y_train):

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight='balanced'))
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        eval_metric='logloss'
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return {
        "logistic": lr,
        "random_forest": rf,
        "xgboost": xgb
    }


def select_best_model(models, X_test, y_test):
    best_model = None
    best_score = 0

    for name, model in models.items():
        preds = model.predict(X_test)
        score = f1_score(y_test, preds)

        print(f"{name} F1 Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    print(f"\nBest model selected with F1: {best_score:.4f}")
    return best_model


def save_model(model, path="outputs/model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")