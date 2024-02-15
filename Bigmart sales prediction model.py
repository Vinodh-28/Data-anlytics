#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the dependencies


# In[2]:


import pandas as pd # for working with dataframes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:


# to read the dataset


# In[68]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data files/Bigmart prediction dataset/Train.csv")


# In[70]:


df.head()


# In[71]:


df.info()


# In[72]:


df.describe()


# In[73]:


df.shape


# In[74]:


df.isnull().sum()


# In[75]:


# inorder to fill up these null values we use mean & mode 
## data preprocessing


# In[78]:


df['Item_Weight'].mean()


# In[80]:


df['Item_Weight'].fillna(df['Item_Weight'].mean(),inplace=True)


# In[81]:


df["Item_Weight"].isnull().sum()


# In[82]:


# to fill up the null values in Outlet Size using model value


# In[83]:


# we use pivot table to fill the null values in Outlet Size


# In[85]:


Outlet_size_mode=df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))


# In[88]:


print(Outlet_size_mode)


# In[90]:


#to fill up the missing values in outlet size


# In[94]:


missing_values=df['Outlet_Size'].isnull()


# In[95]:


missing_values


# In[144]:


df.loc[missing_values,'Outlet_Size']=df.loc[missing_values,'Outlet_Type'].apply(lambda x: Outlet_size_mode.iloc[0][x])


# In[145]:


df['Outlet_Size'].isnull().sum()


# In[146]:


# data analysis


# In[147]:


# univariate analysis


# In[148]:


# distibution of sales among the outlet size and outlet type


# In[149]:


plt.figure(figsize=(10,10))
sns.countplot(x=df['Outlet_Type'])


# In[153]:


sns.countplot(df['Outlet_Size'])


# In[154]:


sns.countplot(df['Outlet_Location_Type'])


# In[162]:


a=sns.countplot(df['Item_Fat_Content'])
plt.ylabel('No. of items ')
plt.show()


# In[163]:


# numerical features 


# In[167]:


# distribution of Item weight 
plt.figure(figsize=(15,10))
sns.distplot(df['Item_Weight'])


# In[168]:


plt.figure(figsize=(15,10))
sns.distplot(df['Item_MRP'])


# In[170]:


sns.countplot(df['Outlet_Establishment_Year'])


# In[172]:


df['Outlet_Establishment_Year'].value_counts()


# In[173]:


#bivariate analysis


# In[193]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True, square=True,cmap=None,fmt='1.1f')


# In[255]:


x=df.groupby(df['Item_Type'],axis=0).mean('Item_MRP')
x


# In[225]:


outlet_age_trend=df.groupby(df['Outlet_Establishment_Year'],axis=0).mean('Item_Outlet_Sales')


# In[226]:


outlet_age_trend


# In[256]:


# data pre processing prior to training the ML model


# In[258]:


# label encode the categorical values


# In[259]:


# Item fat content column 


# In[262]:


df['Item_Fat_Content'].value_counts()


# In[263]:


# standardize the values with same meaning 


# In[268]:


df.replace({'Item_Fat_Content':{'Low Fat': 'LF','LF':'LF','low fat':'LF','reg':'Regular'}},inplace=True)


# In[269]:


df['Item_Fat_Content'].value_counts()


# In[270]:


df.info()


# In[271]:


df['Item_Type'].value_counts()


# In[272]:


df['Outlet_Establishment_Year'].value_counts()


# In[273]:


df['Outlet_Size'].value_counts()


# In[274]:


df['Outlet_Location_Type'].value_counts()


# In[275]:


df['Outlet_Type'].value_counts()


# # label encoding

# In[276]:


# import the encoder function from sklearn module


# In[277]:


from sklearn.preprocessing import LabelEncoder


# In[307]:


encoder=LabelEncoder()


# In[308]:


df['item_Identifier']=encoder.fit_transform(df['Item_Identifier'])


# In[309]:


df.info()


# In[314]:


df['Item_Fat_Content']=encoder.fit_transform(df['Item_Fat_Content'])
df['Item_Type']=encoder.fit_transform(df['Item_Type'])
df['Outlet_Identifier']=encoder.fit_transform(df['Outlet_Identifier'])
df['Outlet_Size']=encoder.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type']=encoder.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type']=encoder.fit_transform(df['Outlet_Type'])
df['Item_Identifier']=encoder.fit_transform(df['Item_Identifier'])


# In[315]:


df


# In[316]:


x=df.drop(['Item_Outlet_Sales','item_Identifier'],axis=1)


# In[317]:


y=df['Item_Outlet_Sales']


# In[318]:


# import the machine learning model & related libraries for training & testing the datasets


# In[319]:


from sklearn.model_selection import train_test_split


# In[320]:


from xgboost import XGBRegressor
from sklearn import metrics


# In[321]:


# spliting the dataset into train and test


# In[322]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[323]:


X_train


# In[324]:


X_test


# In[325]:


regressor=XGBRegressor()


# In[326]:


Model=regressor.fit(X_train,Y_train)


# In[327]:


Y_pred_train=Model.predict(X_train)


# In[328]:


# find the r2 score value to evaluate the accuraccy of the model


# In[329]:


R2_score=metrics.r2_score(Y_train,Y_pred_train)


# In[330]:


R2_score


# In[ ]:




