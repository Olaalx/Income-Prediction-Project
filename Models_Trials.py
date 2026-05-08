import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)




#load dataset
x_train = pd.read_csv("data/processed/x_train.csv")
x_test = pd.read_csv("data/processed/x_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

RESULTS_DIR = Path("results")

#scaling f for logistic + svm

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

def evaluate_model(model, x_tr, y_tr, x_te, y_te, model_name, save_cm_name=None):
    model.fit(x_tr, y_tr)
    preds = model.predict(x_te)

    acc = accuracy_score(y_te, preds)
    report = classification_report(y_te, preds, zero_division=0)
    cm = confusion_matrix(y_te, preds)

    print("\n" + "-" * 55)
    print(model_name.upper())
    print("-" * 55)
    print(f"Accuracy: {acc:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    if save_cm_name:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{model_name} Confusion Matrix")
        plt.savefig(RESULTS_DIR / save_cm_name, bbox_inches="tight", dpi=300)
        plt.close()

    return acc, report, cm


log_model = LogisticRegression(C=1, max_iter=1000, random_state=42)
log_acc, log_report, log_cm = evaluate_model(
    log_model,
    x_train_scaled, y_train,
    x_test_scaled, y_test,
    "Logistic Regression",
    "logistic_confusion_matrix.png"
)

print("\nLogistic Regression Tuning:")
for c in [0.1, 1, 10]:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(x_train_scaled, y_train)
    preds = model.predict(x_test_scaled)
    print(f"C = {c} -> Accuracy = {accuracy_score(y_test, preds):.4f}")


svm_model = SVC(C=1, kernel="rbf", random_state=42)
svm_acc, svm_report, svm_cm = evaluate_model(
    svm_model,
    x_train_scaled, y_train,
    x_test_scaled, y_test,
    "SVM",
    "svm_confusion_matrix.png"
)

print("\nSVM Tuning:")
for c in [0.1, 1, 10]:
    for kernel in ["linear", "rbf"]:
        model = SVC(C=c, kernel=kernel, random_state=42)
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_test_scaled)
        print(f"C = {c}, Kernel = {kernel} -> Accuracy = {accuracy_score(y_test, preds):.4f}")

#DT
dt_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    min_samples_split=20,
    random_state=42
)

dt_acc, dt_report, dt_cm = evaluate_model(
    dt_model,
    x_train, y_train,
    x_test, y_test,
    "Decision Tree",
    "decision_tree_confusion_matrix.png"
)

print("\nDecision Tree Tuning:")
for depth in [5, 10, 15]:
    for split in [2, 10, 20]:
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=depth,
            min_samples_split=split,
            random_state=42
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        print(f"max_depth = {depth}, min_samples_split = {split} -> Accuracy = {accuracy_score(y_test, preds):.4f}")

#xgb
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)

xgb_acc, xgb_report, xgb_cm = evaluate_model(
    xgb_model,
    x_train, y_train,
    x_test, y_test,
    "XGBoost",
    "xgboost_confusion_matrix.png"
)

print("\n       XGBoost Tuning      ")
for depth in [3, 6, 9]:
    for lr in [0.01, 0.1, 0.3]:
        model = XGBClassifier(
            n_estimators=100,
            max_depth=depth,
            learning_rate=lr,
            random_state=42,
            eval_metric="logloss"
        )
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        print(f"max_depth = {depth}, learning_rate = {lr} -> Accuracy = {accuracy_score(y_test, preds):.4f}")

print("\nXGBoost n_estimators Comparison:")
for n in [50, 100, 200]:
    model = XGBClassifier(
        n_estimators=n,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print(f"n_estimators = {n} -> Accuracy = {accuracy_score(y_test, preds):.4f}")


results = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "Decision Tree", "XGBoost"],
    "Accuracy": [log_acc, svm_acc, dt_acc, xgb_acc]
})

print(results)