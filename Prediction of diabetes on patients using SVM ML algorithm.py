#!/usr/bin/env python
# coding: utf-8

# In[1]:


#incidence of diabetes prediction on patients by python


# In[2]:


#import libraries


# In[3]:


import numpy as numpy 
import pandas as pd


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[18]:


#import the dataset (Data collection and analysis)


# In[16]:


df=pd.read_csv('C:/Users/vinodh/OneDrive/Desktop/Data files/diabetes (1).csv')


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[20]:


df.shape


# In[24]:


df['Outcome'].value_counts()


# In[25]:


# 0 - Non diabetic, 1- Diabetic


# In[26]:


df.groupby('Outcome').mean()


# In[28]:


# separate the dataset


# In[29]:


X=df.drop(columns='Outcome',axis=1)


# In[31]:


X.head()


# In[32]:


Y=df['Outcome']


# In[37]:


# data standardization ( Important step in preprocessing )


# In[38]:


scaler=StandardScaler()


# In[41]:


scaler.fit(X)


# In[43]:


Standardized_data = scaler.transform(X)


# In[45]:


X=Standardized_data


# In[47]:


#split the data fot train and test


# In[50]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[51]:


print(X.shape,X_train.shape,X_test.shape)


# In[52]:


# training the model


# In[55]:


classifier=svm.SVC(kernel='linear')


# In[56]:


# training the support vector machine Classifer


# In[58]:


classifier.fit(X_train,Y_train)


# In[59]:


# evaluating the accuracy score of our model


# In[60]:


# checking the accuracy score using the training the data


# In[62]:


X_train_prediction=classifier.predict(X_train)


# In[63]:


training_data_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[65]:


# printing the accuracy score on the traning data
training_data_accuracy


# In[66]:


# predicting the outcomes of the test data sets


# In[67]:


X_test_prediction=classifier.predict(X_test)


# In[69]:


test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[70]:


# accuracy score for the test dataset


# In[71]:


test_data_accuracy


# In[72]:


# building the predictive sysytem


# In[74]:


input_data=(4,110,92,0,0,37.6,0.191,30)


# In[75]:


# changing the input data to a numpy array


# In[78]:


input_array=numpy.asarray(input_data)


# In[79]:


# reshaping the array
reshaped_array=input_array.reshape(1,-1)


# In[80]:


input_array.shape


# In[81]:


reshaped_array.shape


# In[82]:


reshaped_array


# In[84]:


# standardization of reshaped_array


# In[85]:


reshaped_array_std=scaler.fit_transform(reshaped_array)


# In[86]:


std_data_prediction=classifier.predict(reshaped_array_std)


# In[88]:


print(std_data_prediction)


# In[90]:


# model correctly predicted that the person is non diabetic


# In[92]:


if std_data_prediction[0]==0:
    print('person is non diabetic')
else:
    print('person is diabetic')


# In[ ]:




