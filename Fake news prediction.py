#!/usr/bin/env python
# coding: utf-8

# In[1]:


# fake news prediction model


# In[11]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


#import the dataset


# In[13]:


df=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data files/Fake news dataset/train.csv")


# In[14]:


df.head()


# In[15]:


# importing the libraries for machine learning model


# In[16]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[17]:


# to find the stopwords


# In[18]:


import nltk
nltk.download('stopwords')


# In[19]:


print(stopwords.words('english'))  # printing the stopwords in english


# In[20]:


# data collection and data preprocessing 


# In[21]:


df


# In[22]:


df.info()


# In[23]:


df.isnull().sum()


# In[27]:


df.head()


# In[28]:


#replacing the null values with empty string


# In[29]:


df = df.fillna('')


# In[30]:


df.isnull().sum()


# In[31]:


# merging the author name and newstitle name 


# In[42]:


df['Content']= df['author']+' '+df['title']


# In[43]:


df['Content'][0]


# In[44]:


# separating the data and lables


# In[49]:


x=df.drop(columns=['label'],axis=1)


# In[50]:


y=df['label']


# In[51]:


x


# In[48]:


y


# In[53]:


# stemming procedure


# In[54]:


# stemming is the process of reducing the word to its root word


# In[56]:


# example: actor - act, driver - drive


# In[65]:


import re
port_stem = PorterStemmer()


# In[66]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content= stemmed_content.lower()
    stemmed_content= stemmed_content.split()
    stemmed_content= [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content= ' '.join(stemmed_content)
    return stemmed_content


# In[67]:


df['Content']=df['Content'].apply(stemming)


# In[68]:


df['Content']


# In[69]:


# stemming for the content column completed


# In[70]:


X = df['Content'].values


# In[71]:


Y=df['label']


# In[80]:


# converting the textual data to the numerical data


# In[82]:


vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)


# In[85]:


# spliting the dataset to training and test data


# In[90]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[91]:


model=LogisticRegression()


# In[92]:


Regression_model=model.fit(X_train,Y_train)


# In[93]:


Y_pred_train=model.predict(X_train)


# In[94]:


# to find the accuracy score


# In[96]:


accuracy_Score=accuracy_score(Y_pred_train,Y_train)


# In[108]:


print('accuracy score of the test data',accuracy_Score )


# In[103]:


# predicting the values for the test data


# In[104]:


Y_pred = model.predict(X_test)


# In[105]:


accuracy_Score_test = accuracy_score(Y_pred,Y_test)


# In[107]:


print('accuracy score of the test data is ',accuracy_Score_test)


# In[110]:


#Making a prediction system


# In[114]:


print(df['Content'][0])


# In[129]:


X_new= X_test[0]


# In[137]:


prediction=model.predict(X_new)

print(prediction)

if prediction==0:
    print('the news is real')
    
else:
    print('The news is fake')


# In[138]:


prediction=model.predict(X_test[1])

print(prediction)

if prediction==0:
    print('the news is real')
    
else:
    print('The news is fake')


# In[144]:


print(X_train[0])


# In[ ]:




