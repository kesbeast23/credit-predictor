import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("german_credit_data.csv")
print(df.info())
#Looking unique values
print(df.nunique())
#Looking the data
print(df.head())
df.replace(to_replace ="good", 
                            value =1) 
include = ['Age', 'Sex', 'Job', 'Housing','Saving accounts','Checking account','Credit amount','Duration','Purpose','Risk'] # Only four features
df_ = df[include]
categoricals = []
for col, col_type in df_.dtypes.iteritems():
     if col_type == 'O':
          categoricals.append(col)
     else:
          df_[col].fillna(0, inplace=True)

df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Risk'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x, y)

from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')

# Load the model that  just saved
lr = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")