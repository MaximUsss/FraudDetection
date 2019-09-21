from sklearn import preprocessing
import pandas as pd
from fraud_detection_model import buildModel
import numpy as np

# number of rows in data set
# used for testing purposes
# n = 1000

# data from kaggle IEEE-CIS Fraud Detection
# https://www.kaggle.com/c/ieee-fraud-detection/data
# Read the CSV file
identity = pd.read_csv("J:/Study/ML/Siraj/Homework/Week 3/data/train_identity.csv")
transaction = pd.read_csv("J:/Study/ML/Siraj/Homework/Week 3/data/train_transaction.csv",)
test_identity = pd.read_csv("J:/Study/ML/Siraj/Homework/Week 3/data/test_identity.csv",)
test_transaction = pd.read_csv("J:/Study/ML/Siraj/Homework/Week 3/data/test_transaction.csv")

# let's combine the data and work with the whole dataset
train = pd.merge(transaction, identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# define if there are columns with more than 90% nulls
many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]

# define columns with only 1 value
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

# define top value columns
big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

# merging useless columns in a list
cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove('isFraud')

# dropping useless columns
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# Drop missing values
train.fillna(value=-99999, inplace=True)

# converting the labels into numeric form with LabelEncoder class
# so as to convert it into the machine-readable form
cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3', 'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']
for col in cat_cols:
    if col in train.columns:
        le = preprocessing.LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))

X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']

# looks like dataset is missing "isFraud" label in test data
# X_test = test.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
# y_test = test.sort_values('TransactionDT')['isFraud']


# method from cleaning infinite values to NaN
def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)


# Cleaning infinite values to NaN
X = clean_inf_nan(X)
# X_test = clean_inf_nan(X_test)

# Scale the X so that everyone can have the same distribution
# https://scikit-learn.org/stable/modules/preprocessing.html
X = preprocessing.scale(X)
# X_test = preprocessing.scale(X_test)

buildModel(X, y)
