#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the dependencies 


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# importing the dataset


# In[4]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data analyst files/heart_disease_data.csv")


# In[5]:


df.head() # print the first 5 rows of the dataframe 


# In[6]:


df.info()


# In[8]:


df.isnull().sum() # to check for the missing values 


# In[9]:


df.describe()


# In[10]:


# splitting the data into features and target values


# In[11]:


x=df.drop(['target'],axis=1)


# In[16]:


x.shape


# In[17]:


y=df.iloc[:,-1]


# In[18]:


y.shape


# In[19]:


# splitting the data into test and train 


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[25]:


from sklearn.metrics import accuracy_score


# In[26]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[29]:


X_test


# In[30]:


# create a model object


# In[31]:


Regressor=LogisticRegression()


# In[32]:


Model=Regressor.fit(X_train,Y_train)


# In[34]:


# to predict the outcome for the test dataset


# In[35]:


# to evaluate the accuracy score based on the training dataset


# In[36]:


Y_pred_train=Model.predict(X_train)


# In[37]:


accuracy_score_train=accuracy_score(Y_pred_train,Y_train)


# In[38]:


print('The accuracy score of the training dataset is ',accuracy_score_train)


# In[39]:


# to evaluate the accuracy score on the test dataset 


# In[40]:


Y_pred_test=Model.predict(X_test)


# In[41]:


accuracy_score_test=accuracy_score(Y_pred_test,Y_test)


# In[42]:


print('The accuracy score of model on the test dataset is ',accuracy_score_test)


# In[ ]:




