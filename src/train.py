import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("data/train.csv")

X = df.drop("price_range", axis=1)
y = df["price_range"]

# -----------------------------
# Train-validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("mobile-price-lightweight")

best_model = None
best_acc = 0
best_model_name = ""

# -----------------------------
# 1. Logistic Regression (Scaled)
# -----------------------------
with mlflow.start_run(run_name="logistic_regression_scaled"):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train_scaled, y_train)

    preds = lr.predict(X_val_scaled)
    acc = accuracy_score(y_val, preds)

    mlflow.log_param("model", "LogisticRegression_scaled")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(lr, "model")

    print(f"LR (Scaled) Accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model = (lr, scaler)   # store scaler also
        best_model_name = "logistic_regression_scaled"

# -----------------------------
# 2. Decision Tree (Lightweight)
# -----------------------------
with mlflow.start_run(run_name="decision_tree_light"):

    dt = DecisionTreeClassifier(
        max_depth=3,
        min_samples_split=10,
        random_state=42
    )

    dt.fit(X_train, y_train)

    preds = dt.predict(X_val)
    acc = accuracy_score(y_val, preds)

    mlflow.log_param("model", "DecisionTree")
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(dt, "model")

    print(f"DT Accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_model = (dt, None)   # no scaler needed
        best_model_name = "decision_tree"

# -----------------------------
# Save BEST model
# -----------------------------
model, scaler = best_model

joblib.dump(model, "models/model.pkl")

# Save scaler only if exists
if scaler is not None:
    joblib.dump(scaler, "models/scaler.pkl")

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_acc}")
print("==============================")
print("model saved")
print("done")
print("done")
print("done")
print("=====")
 