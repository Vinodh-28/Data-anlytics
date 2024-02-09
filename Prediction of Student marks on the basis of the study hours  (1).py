#!/usr/bin/env python
# coding: utf-8

# # Prediction of Marks of the student based on the study hours
# AUTHOR: VINODH                                                                 

# In[1]:


# importing the dependencies


# In[5]:


import pandas as pd  # for working with the dataframe 
import numpy as np # for using numerical functions
import matplotlib.pyplot as plt  # for visualization 
import seaborn as sns # for adavanced univariate variation visualization 


# In[4]:


# import the dataset


# In[6]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data analyst files/Student_marks.csv")


# In[8]:


df.head()  # chekcing the first 5 entries of the dataset


# In[9]:


df.shape   # chekcing the shape of the dataframe 


# In[10]:


df.describe()  # checking the Statistical measures of the column attributes of the dataset


# In[11]:


df.info()  # checking the info related to column attributes


# In[12]:


#dividiing the data into independent features and labels


# In[49]:


x=df.drop(['Scores'],axis=1)
x.shape


# In[51]:


y=df.drop(['Hours'],axis=1)
y.shape


# In[52]:


x.shape


# In[53]:


y.shape


# In[54]:


# importing the libraries for building and training the model


# In[55]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[56]:


from sklearn import metrics


# In[57]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[58]:


regressor=LinearRegression()  # creating the model object


# In[61]:


model=regressor.fit(X_train,Y_train)


# In[65]:


# evaluating the accuracy of model 
Y_pred_train=model.predict(X_train)


# In[67]:


R2_score=metrics.r2_score(Y_pred_train,Y_train)


# In[68]:


print(R2_score)


# In[69]:


# prediction of values using the test dataset


# In[70]:


Y_test_pred=model.predict(X_test)


# In[71]:


R2_score_test=metrics.r2_score(Y_test_pred,Y_test)


# In[72]:


print(R2_score_test)


# prediction of the student's marks score for the 9.25 hours/day of study time

# In[77]:


marks_pred=model.predict([[9.25]])


# In[81]:


print(marks_pred[0][0])


# In[82]:


print('Marks of student who study 9.25 hours/ day is',marks_pred[0][0])


# In[ ]:




