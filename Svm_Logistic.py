import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================
# LOAD DATA
# =========================
# Make sure Member 1 provides this file
df = pd.read_csv("data/processed/cleaned_data.csv")

# =========================
# FEATURES & TARGET
# =========================
X = df.drop("income", axis=1)
y = df["income"]

# =========================
# TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================
# LOGISTIC REGRESSION
# =========================
print("===== Logistic Regression =====")

log_model = LogisticRegression(C=1, max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


# =========================
# LOGISTIC HYPERPARAMETER TUNING
# =========================
print("\n===== Logistic Regression Tuning =====")

for c in [0.1, 1, 10]:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"C = {c}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("-" * 30)


# =========================
# SVM MODEL
# =========================
print("\n===== SVM =====")

svm_model = SVC(C=1, kernel='rbf')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# =========================
# SVM HYPERPARAMETER TUNING
# =========================
print("\n===== SVM Tuning =====")

for c in [0.1, 1, 10]:
    for kernel in ['linear', 'rbf']:
        model = SVC(C=c, kernel=kernel)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print(f"C = {c}, Kernel = {kernel}")
        print("Accuracy:", accuracy_score(y_test, preds))
        print("-" * 40)