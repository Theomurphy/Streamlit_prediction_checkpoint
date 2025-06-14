import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.linear_model import LogisticRegression
import joblib
import streamlit as st 

# Import the data 

path = 'C:/Users/Trois/Downloads/Video/Data Science/Month 4/Datasets/Financial_inclusion_dataset.csv'

data = pd.read_csv(path)

data.head()

# Data info 

data.info()

data.describe()

# Data profiling 

#rapport = ProfileReport(data)

#rapport.to_notebook_iframe()

# Heatmap 

data_heatmap = data.select_dtypes(exclude='object')

sns.heatmap(data_heatmap.corr(), annot=True)
plt.show()


# Remove irrelevant columns 

data.drop(columns='uniqueid', inplace=True)


# Check for missing values 

data.isna().sum()

# Check for duplicate

data.duplicated().sum()

# Drop Duplicate

data.drop_duplicates()

# Encoding the data

encoder = LabelEncoder()

cat_col = data.select_dtypes(include='object')

for col in cat_col:
    data[col] = encoder.fit_transform(data[col])
    
# Defining the input and output data

x = data.drop(columns='bank_account')

y = data['bank_account']

# Spliting the data 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 33 )

# Training the model 

model = LogisticRegression()

model.fit(x_train, y_train)

# Predicttion 
predict = model.predict(x_test)

#Evaluate the model 
accuracy_score(predict, y_test)


# Save the model to a file

joblib.dump(model, 'LR_model.pkl') 


