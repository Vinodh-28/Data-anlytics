#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To do exploratory data analysis on retail dataset


# In[2]:


#importing the librabries


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


#importing the dataset
df=pd.read_csv('C:/Users/vinodh/OneDrive/Desktop/Data files/Sparks task datasets/SampleSuperstore (2).csv')


# In[6]:


df.head()


# In[7]:


df.info()


# In[10]:


df.describe() # to identify the statistical measures of numerical attributes of the dataset


# In[12]:


df.shape # to find the number of entries & no of attributes for every entry


# In[13]:


df.size


# In[14]:


# data preprocessing


# In[15]:


df.isnull().sum()


# In[16]:


#univariate data analysis


# In[17]:


#to identify the maximum used ship mode


# In[18]:


sns.countplot(df['Ship Mode'])


# In[19]:


# maximum used ship mode is standard class


# In[23]:


#to identify maximum purchased segment


# In[22]:


df.nunique()


# In[24]:


sns.countplot(df['Segment'])


# In[25]:


#consumer is the maximum purchasing customer segment


# In[62]:


#to find the region which orders maximum number of products


# In[61]:


sns.countplot(df['Region'])


# In[72]:


plt.figure(figsize=(20,8))
sns.countplot(df['Sub-Category'])


# In[73]:


#binders is the maximum purchased sub categoty in the store


# In[82]:


df.groupby('Region')['Sales'].sum().plot.bar()


# In[84]:


# west region has contributed maximum sales in the last year


# In[116]:


# to identify the region which contributed maximum profit 
a=df.groupby('Region')['Profit'].sum().plot.bar()
plt.xlabel('Region')
plt.ylabel('Profit')


# In[117]:


# west region has contributed maximum profit in the last year


# In[118]:


# to identify the frequently given discount percentage


# In[119]:


plt.figure(figsize=(10,10))
sns.distplot(df['Discount'])


# In[144]:



sns.boxplot(df['Discount'])


# In[ ]:




