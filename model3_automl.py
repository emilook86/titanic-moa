from pathlib import Path
import pandas as pd

try:
    from flaml.automl import AutoML
except ImportError:
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: flaml. Install with `pip install flaml` and retry."
        ) from exc


PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_PATH = OUTPUT_DIR / "automl_predictions.csv"

X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
X_test_path = PROCESSED_DATA_DIR / "X_test.csv"

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)

train_ids = X_train["PassengerId"]
test_ids = X_test["PassengerId"]

X_train = X_train.drop(columns=["PassengerId"])
X_test = X_test.drop(columns=["PassengerId"])

y_train = y_train.iloc[:, 0]

automl = AutoML()
automl_settings = {
    "time_budget": 3000,
    "metric": "accuracy",
    "task": "classification",
    "eval_method": "cv",
    "n_splits": 5,
    "seed": 42,
    "retrain_full": True,
    "log_file_name": str(OUTPUT_DIR / "flaml_automl.log"),
}

automl.fit(X_train, y_train, **automl_settings)
y_pred = automl.predict(X_test).astype(bool)
submission = pd.DataFrame({"PassengerId": test_ids, "Transported": y_pred})
submission.to_csv(PREDICTIONS_PATH, index=False)

print("Predictions saved to:", PREDICTIONS_PATH)
print("=" * 50)
print("BEST MODEL INFORMATION")
print("=" * 50)
print(f"\nBest estimator: {automl.best_estimator}")
print("Best config:")
for param, value in automl.best_config.items():
    print(f"  {param}: {value}")

best_loss = getattr(automl, "best_loss", None)
best_score = getattr(automl, "best_score", None)
metric_name = automl_settings.get("metric")

if best_score is None and best_loss is not None and metric_name == "accuracy":
    best_score = 1 - best_loss

print(f"\nBest CV score: {best_score:.4f}")
print(f"Best loss: {best_loss:.4f}")

# ==================================================
# BEST MODEL INFORMATION
# ==================================================

# Best estimator: xgboost
# Best config:
#   n_estimators: 283
#   max_leaves: 8
#   min_child_weight: 0.24548583393846193
#   learning_rate: 0.06456119721207722
#   subsample: 0.9509358747888187
#   colsample_bylevel: 0.7646612217167584
#   colsample_bytree: 0.8516012204569093
#   reg_alpha: 0.00117192269788344
#   reg_lambda: 0.0019807174167241004

# Best CV score: 0.8132
# Best loss: 0.1868
