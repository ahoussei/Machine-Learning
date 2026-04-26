#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# In[2]:


train_df = pd.read_csv("train.csv")
train_df.info()


# In[3]:


train_df.head()


# In[4]:


train_df.describe()


# In[5]:


test_df = pd.read_csv("test.csv")
test_df.info()


# In[6]:


test_df.head()


# In[7]:


# filter native country for only US results
# get rid of capital gain, capital loss, and fnlwgt columns
# fill any null values with "unknown" (safety net, although there do not seem to be any missing)
def clean_data(df):
    df = df[df["native-country"] == "United-States"]
    
    df = df.drop(columns=["fnlwgt", "capital-gain", "capital-loss"])
    
    df["workclass"] = df["workclass"].fillna("Unknown")
    df["occupation"] = df["occupation"].fillna("Unknown")
    
    return df

train_df = clean_data(train_df)
test_df = clean_data(test_df)


# In[8]:


train_df.head()


# In[9]:


test_df.head()


# In[10]:


# encoding 
train_df_encoding = pd.get_dummies(train_df, columns= ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'], dtype=int)
test_df_encoding = pd.get_dummies(test_df, columns= ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'], dtype=int)
train_df_encoding.head()
test_df_encoding.head()
train_df_encoding.info()
test_df_encoding.info()


# In[11]:


train_df_encoding, test_df_encoding = train_df_encoding.align(
    test_df_encoding, join='left', axis=1, fill_value=0)


# In[12]:


set(train_df_encoding.columns)==set(test_df_encoding.columns)


# In[13]:


numeric_cols = ['age', 'hours-per-week']
scaler = StandardScaler()
# fitting the scaler only to the train data
scaler.fit(train_df[numeric_cols])
# transforming both train and test
train_df_encoding[numeric_cols] = scaler.transform(train_df_encoding[numeric_cols])
test_df_encoding[numeric_cols] = scaler.transform(test_df_encoding[numeric_cols])
# print
print("Scaled train head:", train_df_encoding[numeric_cols].head())
print("Scaled train describe:", train_df_encoding[numeric_cols].describe())


# In[17]:


# defining features and target for education
EDU_FEATURES = [col for col in train_df_encoding.columns
                if any(col.startswith(p) for p in
                      ['age', 'workclass', 'occupation', 'income',
                'marital', 'relationship', 'race', 'gender', 'hours'])]
EDU_TARGET = 'educational-num'

# splitting the data
X_edu = train_df_encoding[EDU_FEATURES]
y_edu = train_df_encoding[EDU_TARGET]

X_edu_train, X_edu_dev, y_edu_train, y_edu_dev = train_test_split(
    X_edu, y_edu, test_size=0.15, random_state=42)

print(X_edu_train.shape)
print(X_edu_dev.shape)


# In[30]:


bins = [0, 8, 12, 14, 17]
labels = [0, 1, 2, 3]  # 0=Low, 1=HS-level, 2=Some-college, 3=Advanced

y_edu_train_binned = pd.cut(y_edu_train, bins=bins, labels=labels).astype(int)
y_edu_dev_binned = pd.cut(y_edu_dev, bins=bins, labels=labels).astype(int)


# In[34]:


#Model selection and prediction 

#Decision Tree
decision_tree= DecisionTreeClassifier(random_state=42, max_depth=10)

#fitting the classifier
decision_tree.fit(X_edu_train,y_edu_train_binned)

#making predictions
edu_y_prediction = decision_tree.predict(X_edu_dev)
edu_train_score = decision_tree.score(X_edu_train, y_edu_train_binned)
edu_dev_score = decision_tree.score(X_edu_dev, y_edu_dev_binned)
precision = precision_score(y_edu_dev_binned, edu_y_prediction,average='micro')
recall = recall_score(y_edu_dev_binned, edu_y_prediction, average='micro')

print(edu_y_prediction,edu_train_score,edu_dev_score, precision, recall)


# In[ ]:




