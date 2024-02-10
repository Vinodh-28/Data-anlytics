#!/usr/bin/env python
# coding: utf-8

# # IDENTIFY THE OPTIMUM NUMBER OF CLUSTERS USING UNSUPERVISED ML & VISUALIZING THEM 
# 
# AUTHOR : VINODH M

# In[ ]:


#importing the relevant libraries # predicting the species 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


#importing the dataset
iris=pd.read_csv("C:/Users/vinodh/OneDrive/Desktop/Data files/Sparks task datasets/Iris (1).csv")


# In[17]:


type(iris)


# In[18]:


iris.shape


# In[19]:


iris.size


# In[20]:


iris.info()


# In[21]:


iris.describe()


# In[23]:


iris.head()


# In[56]:


iris.isnull().sum()


# In[67]:


iris.iloc[:,5].nunique()


# In[69]:


iris.iloc[:,5].value_counts()


# In[87]:


import seaborn as sns
#visualization


# In[88]:


sns.set(style='whitegrid')
ax=sns.stripplot(x='Species',y='SepalLengthCm',data=iris)
plt.title('Iris datasets')
plt.show()


# In[92]:


a=sns.boxplot(x='Species',y='SepalWidthCm',data=iris)

plt.show(a)


# In[93]:


b=sns.boxplot(x='Species',y='PetalLengthCm',data=iris)
plt.show(b)


# In[94]:


c=sns.boxplot(x='Species',y='PetalWidthCm',data=iris)


# In[98]:


#heatmap
sns.heatmap(iris.corr(), annot=True)
plt.title('heatmap')
plt.show()


# In[99]:


#finding the optimum number of clusters for K means clsutering algorithm 


# In[117]:


x=iris.iloc[:,[1,2,3,4]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(init='k-means++',n_clusters=i,n_init=10,max_iter=50,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    print('k:',i, 'wcss=',kmeans.inertia_)


# In[118]:


#plotting the results to find the optimum number of clusters using elbow method


# In[121]:


plt.plot(range(1,11),wcss)
plt.xlabel('K values')
plt.ylabel('wcss')
plt.title('Wcss vs K values')
plt.show()


# In[123]:


#from here we can infer that the K = 3 is the optimum number of clusters for this dataset


# In[124]:


kmeans=KMeans(init='k-means++',n_clusters=3,n_init=10,max_iter=50,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# In[125]:


#visualizing the clusters


# In[129]:


len(y_kmeans)


# In[150]:


x[y_kmeans==2,1].shape


# In[151]:


plt.figure(figsize=(5,5))
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Iris_setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='green',label='Iris-Versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='blue',label='Iirs-Virginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title('Iris flower Cluster')
plt.xlabel('Sepal length on Cm')
plt.ylabel('petal length in cm')
plt.legend()
plt.show()


# In[153]:


plt.figure(figsize=(5,5))
plt.scatter(x[y_kmeans==1,1],x[y_kmeans==1,2],s=100,c='red',label='Iris_setosa')
plt.scatter(x[y_kmeans==2,1],x[y_kmeans==2,2],s=100,c='green',label='Iris-Versicolour')
plt.scatter(x[y_kmeans==3,1],x[y_kmeans==3,2],s=100,c='blue',label='Iirs-Virginica')

plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],s=300,c='yellow',label='centroids')


# In[ ]:




