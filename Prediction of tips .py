#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


# In[3]:


tips = sns.load_dataset('tips')


# In[4]:


tips.head()


# In[7]:


sns.set_theme()
sns.relplot(x='total_bill',y='tip',data=tips,col='time',hue='smoker',style='smoker',size='size')


# In[8]:


#load the iris dataset


# In[9]:


iris= sns.load_dataset('iris')
iris.head()


# In[11]:


sns.scatterplot(x='sepal_length',y='sepal_width',hue='species',data=iris)


# In[12]:


sns.scatterplot(x='sepal_length',y='petal_width',hue='species',data=iris)


# In[13]:


titanic=sns.load_dataset('titanic')


# In[14]:


titanic.head()


# In[16]:


sns.countplot(data=titanic,x='class')


# In[20]:


titanic.shape


# In[21]:


titanic.size


# In[22]:


titanic.describe()


# In[23]:


sns.countplot(data=titanic,x='survived')


# In[24]:


#barplot


# In[25]:


sns.barplot(x='sex',y='survived',hue='class',data=titanic)


# In[26]:


#house price dataset


# In[34]:


from sklearn.datasets import load_boston
house_boston = load_boston()


# In[39]:


house=pd.DataFrame(house_boston.data, columns=house_boston.feature_names)
house['PRICE']=house_boston.target


# In[40]:


house_boston


# In[41]:


house.head()


# In[44]:


sns.distplot(house['PRICE'])

# heat map
# In[45]:


correlation= house.corr()


# In[59]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f', annot=True,annot_kws={'size':10},cmap=None)


# In[51]:




