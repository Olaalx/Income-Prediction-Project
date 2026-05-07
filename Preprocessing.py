import pandas as pd
import numpy as np

# Understand Columns
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')
print(train_df.head())
print(test_df.head())

# Handle missing values
print(test_df.columns)
print(train_df.columns)
train_df = train_df.rename(columns={'Income ':'Income'})
test_df = test_df.rename(columns={'Income ':'Income'})
columns_to_clean = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'Income']
for col in columns_to_clean:
    train_df[col] = train_df[col].str.strip()
    test_df[col] = test_df[col].str.strip()
train_df = train_df.replace('?', np.nan)
test_df = test_df.replace('?', np.nan)    # 32562 train
print(train_df.isnull().sum())                     # 16282 test
print(test_df.isnull().sum())
colums_with_nan = ['workclass', 'occupation', 'native-country']
for col in colums_with_nan:
    most_freq = train_df[col].mode()[0]
    train_df[col] = train_df[col].fillna(most_freq)
    test_df[col] = test_df[col].fillna(most_freq)
print(train_df.isnull().sum())
print(test_df.isnull().sum())

# Remove duplicates
print(train_df.duplicated().sum())
print(test_df.duplicated().sum())
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
print(train_df.duplicated().sum())
print(test_df.duplicated().sum())

# Basic data cleaning
print(train_df.info())
print(test_df.info())
test_df['Income'] = test_df['Income'].str.replace('.', '')
print(test_df['Income'].unique())
print(train_df['Income'].unique())

# Encode categorical data & Consistency
num_train = train_df.shape[0]
all_data = pd.concat([train_df, test_df], axis=0)
all_data_encoded = pd.get_dummies(all_data)
train_df_final = all_data_encoded.iloc[:num_train, :]
test_df_final = all_data_encoded.iloc[num_train:, :]
print(train_df_final.columns)
y_train = train_df_final['Income_>50K']
y_test = test_df_final['Income_>50K']
x_train = train_df_final.drop(['Income_<=50K', 'Income_>50K'], axis=1)
y_test = test_df_final.drop(['Income_<=50K', 'Income_>50K'], axis=1)