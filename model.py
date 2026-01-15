from xgboost import XGBClassifier
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path.cwd()
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
X_test_path  = PROCESSED_DATA_DIR / "X_test.csv"

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test  = pd.read_csv(X_test_path)

train_ids = X_train['PassengerId']
test_ids  = X_test['PassengerId']

X_train = X_train.drop(columns=['PassengerId'])
X_test  = X_test.drop(columns=['PassengerId'])

param_grid = {
    'max_depth': [3,5,7],
    'n_estimators': [100,200,300],
    'learning_rate': [0.01,0.1,0.2]
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="error"),
    param_grid,
    scoring="accuracy",
    cv=5
)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test).astype(bool)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": y_pred
})

submission.to_csv(OUTPUT_DIR / "xgb_predictions.csv", index=False)

if __name__ == '__main__':
    print("Predictions saved to:", OUTPUT_DIR / "xgb_predictions.csv")