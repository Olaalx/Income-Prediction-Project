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
📊 Results Summary (Example Format)
Model	Accuracy	F1 Score
Logistic Regression	85%	0.84
SVM	87%	0.86
Decision Tree	83%	0.82
Random Forest ⭐	89%	0.88
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
