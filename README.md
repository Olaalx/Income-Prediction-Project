ncome Prediction Using Machine Learning
🧠 Project Overview

This project builds a machine learning system to predict whether an individual's income is greater than $50K or less than or equal to $50K based on demographic and work-related attributes.

We apply multiple classification models and analyze their performance using preprocessing, visualization, and hyperparameter tuning.

📁 Dataset Description

The dataset contains demographic and employment-related features such as:

Age
Workclass
Education
Marital Status
Occupation
Race
Sex
Hours per week
Native country
Income (Target)
🎯 Target Variable:
<=50K → income less than or equal to $50,000
>50K → income greater than $50,000
⚙️ Project Pipeline
1. Data Preprocessing
Handling missing values (? → NaN)
Imputing missing data (mode/median)
Encoding categorical variables (One-Hot Encoding)
Feature scaling (StandardScaler)
Train-test consistency handling
2. Exploratory Data Analysis (EDA)

We analyzed relationships between features using:

Income distribution plots
Age vs Income
Education vs Income
Work hours impact
Correlation heatmap
3. Machine Learning Models

We trained and evaluated the following models:

🔵 Logistic Regression
Hyperparameters: C, penalty
🟢 Support Vector Machine (SVM)
Hyperparameters: C, kernel
🔴 Decision Tree Classifier
Hyperparameters: max_depth, min_samples_split
⭐ Bonus Model (Optional)
Random Forest / XGBoost
📈 Model Evaluation Metrics

We evaluate models using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
📊 Results Summary :

===== Logistic Regression =====
Accuracy: 0.8510076185795036
              precision    recall  f1-score   support

       False       0.88      0.93      0.91     12430
        True       0.73      0.59      0.65      3846

    accuracy                           0.85     16276
   macro avg       0.80      0.76      0.78     16276
weighted avg       0.84      0.85      0.85     16276


===== Logistic Regression Tuning =====
C = 0.1
Accuracy: 0.8510690587367904
------------------------------
C = 1
Accuracy: 0.8510076185795036
------------------------------
C = 10
Accuracy: 0.8508847382649299
------------------------------

===== SVM =====
Accuracy: 0.8527893831408209
              precision    recall  f1-score   support

       False       0.88      0.94      0.91     12430
        True       0.74      0.58      0.65      3846

    accuracy                           0.85     16276
   macro avg       0.81      0.76      0.78     16276
weighted avg       0.85      0.85      0.85     16276


===== SVM Tuning =====
C = 0.1, Kernel = linear
Accuracy: 0.8526665028262472
----------------------------------------
C = 0.1, Kernel = rbf
Accuracy: 0.8429589579749324
----------------------------------------
C = 1, Kernel = linear
Accuracy: 0.8524207421971001
----------------------------------------
C = 1, Kernel = rbf
Accuracy: 0.8527893831408209
----------------------------------------
C = 10, Kernel = linear
Accuracy: 0.8523593020398132
----------------------------------------
C = 10, Kernel = rbf
Accuracy: 0.8497788154337675
----------------------------------------


🔧 Hyperparameter Tuning Analysis

We studied the effect of key hyperparameters:

Logistic Regression (C):
Low C → stronger regularization → simpler model
High C → weaker regularization → better training fit but risk of overfitting
SVM (C, Kernel):
Linear kernel works well for simple separation
RBF kernel captures non-linear patterns
Decision Tree (max_depth):
Small depth → underfitting
Large depth → overfitting

📊 Visualizations

The project includes:

Income distribution plots
Feature vs income analysis
Correlation heatmap
Model comparison charts

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
