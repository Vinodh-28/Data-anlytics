#!/usr/bin/env python
# coding: utf-8

# In[1]:


# improting the libraries


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# impoting the datasets


# In[5]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data files/creditcard.csv")


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.head()


# In[11]:


df['Class'].value_counts()


# In[20]:


df.describe()


# In[21]:


# this dataset is highly unbalanced


# In[22]:


# separating the dataset for analysis


# In[33]:


legit =df[df.Class ==0]
fraud =df[df.Class ==1]


# In[34]:


legit.shape


# In[35]:


fraud.shape


# In[36]:


# analysing the amount field in legit dataset


# In[37]:


legit.Amount.describe()


# In[38]:


fraud.Amount.describe()


# In[39]:


#compare the values for both fraud and legit transcation


# In[40]:


df.groupby('Class').mean()


# In[41]:


# dealing with unbalanced dataset


# In[44]:


# undersampling - build a sample dataset which contains the same proportion of both normal and fraudulent transactions


# In[45]:


legit_sample=legit.sample(n=492)


# In[46]:


legit_sample.head()


# In[47]:


# concatenation of both fraud and legit transaction 


# In[48]:


new_df=pd.concat([legit_sample,fraud],axis=0)


# In[49]:


new_df.shape


# In[50]:


new_df['Class'].value_counts()


# In[51]:


new_df.tail()


# In[55]:


new_df.groupby('Class').mean()


# In[56]:


# divide/split the data into features and target


# In[60]:


X=new_df.drop(['Class'],axis=1)


# In[61]:


Y=new_df['Class']


# In[65]:


X


# In[64]:


Y


# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[73]:


# spliting the dataset into train and test dataset


# In[82]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[83]:


#create model object and train the model


# In[84]:


Regressor=LogisticRegression()


# In[85]:


Regressor=Regressor.fit(X_train,Y_train)


# In[86]:


Y_train


# In[88]:


#evaluating the accuracy of the model


# In[89]:


Y_train_pred=Regressor.predict(X_train)


# In[90]:


Score=accuracy_score(Y_train_pred,Y_train)


# In[93]:


print(Score)


# In[94]:


# to predit the test datset and evaluate based on the test dataset


# In[95]:


Y_pred=Regressor.predict(X_test)


# In[96]:


test_dataset_Score=accuracy_score(Y_pred,Y_test)


# In[97]:


print(test_dataset_Score)


# In[98]:


# visualizing the prediction and test data sets


# In[114]:


Y_test.shape


# In[124]:


Y_pred.shape


# In[ ]:




