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
    "time_budget": 300,
    "metric": "accuracy",
    "task": "classification",
    "eval_method": "cv",
    "n_splits": 5,
    "seed": 42,
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

# Best estimator: lgbm
# Best config:
#   n_estimators: 662
#   num_leaves: 7
#   min_child_samples: 71
#   learning_rate: 0.5992896694559361
#   log_max_bin: 6
#   colsample_bytree: 0.9837397563360719
#   reg_alpha: 0.02287618842773019
#   reg_lambda: 1024.0

# Best CV score: 0.8077
# Best loss: 0.1923
