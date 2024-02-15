#!/usr/bin/env python
# coding: utf-8

# In[1]:


# prediction of gold prices using linear regression model


# In[2]:


# importing the libraries


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


#importing the dataset


# In[6]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data files/gld_price_data.csv")


# In[7]:


df.head()


# In[8]:


#import libraries for machine learning 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics


# In[9]:


df.info()


# In[11]:


df.tail()


# In[12]:


df.shape


# In[13]:


df.size


# In[21]:


df.describe()


# In[23]:


sns.distplot(df['SPX'])


# In[24]:


sns.distplot(df['GLD'])


# In[25]:


# finding the corelation between the varibales


# In[27]:


correlation = df.corr()


# In[29]:


# construction of heatmap for the corelation of the varibales


# In[35]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True,fmt='1.1f',annot=True,annot_kws={'size':15},cmap='Blues')


# In[36]:


print(correlation['GLD'])


# In[37]:


sns.distplot(df['GLD'])


# In[38]:


# removing the date column & spliting the data for modelling and prediction


# In[52]:


x=df.drop(['GLD','Date'],axis=1)


# In[53]:


x


# In[54]:


y=df['GLD']


# In[55]:


y


# In[56]:


#splitting the data into the train and test


# In[57]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[63]:


gld_data_regressor=RandomForestRegressor(n_estimators=100)


# In[64]:


regressor_model=gld_data_regressor.fit(x_train,y_train)


# In[66]:


#prediciton & evaluation of the accuracy of the model


# In[68]:


y_pred=gld_data_regressor.predict(x_test)


# In[70]:


error_score=metrics.r2_score(y_pred,y_test)


# In[71]:


error_score


# In[72]:


# to check visually the accuracy of the model


# In[75]:


sns.scatterplot(y_pred,y_test)
plt.xlabel('predicted value')
plt.ylabel('actual value')
plt.title('accuracy of the predition of model')


# In[ ]:




