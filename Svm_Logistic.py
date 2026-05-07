import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report



x_train = pd.read_csv("data/processed/x_train.csv")
x_test = pd.read_csv("data/processed/x_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

y_train = y_train.squeeze()
y_test = y_test.squeeze()

#scaling first
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


print("===== Logistic Regression =====")

log_model = LogisticRegression(C=1, max_iter=1000)

log_model.fit(x_train, y_train)

log_preds = log_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

print("\n===== Logistic Regression Tuning =====")

for c in [0.1, 1, 10]:

    model = LogisticRegression(C=c, max_iter=1000)

    model.fit(x_train, y_train)

    preds = model.predict(x_test)

    print(f"C = {c}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("-" * 30)


print("\n===== SVM =====")

svm_model = SVC(C=1, kernel='rbf')

svm_model.fit(x_train, y_train)

svm_preds = svm_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, svm_preds))
print(classification_report(y_test, svm_preds))


print("\n===== SVM Tuning =====")

for c in [0.1, 1, 10]:

    for kernel in ['linear', 'rbf']:

        model = SVC(C=c, kernel=kernel)

        model.fit(x_train, y_train)

        preds = model.predict(x_test)

        print(f"C = {c}, Kernel = {kernel}")
        print("Accuracy:", accuracy_score(y_test, preds))
        print("-" * 40)