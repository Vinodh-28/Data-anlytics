#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df=pd.read_csv('C:/Users/vinodh/Downloads/train_ctrUa4K.csv')


# In[8]:


df.head()


# In[9]:


df.shape


# In[19]:


df.describe()


# In[20]:


df.isnull().sum()


# In[21]:


#handling missing values


# In[24]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)


# In[27]:


#data analysis and visualization


# In[29]:


#univariate and bivariate Analysis


# In[32]:


import seaborn as sns


# In[33]:


sns.set()


# In[34]:


#univariate


# In[82]:


plt.figure(figsize=([10,6]))
plt.xlabel=('ApplicantIncome')
plt.title=('Applicant Income vs count')
sns.distplot(df['ApplicantIncome'],color='Green')


# In[84]:


df['ApplicantIncome'].plot.box(figsize=(16,5))
plt.show()


# In[86]:


df.boxplot(column=['ApplicantIncome'],by='Education')
plt.title='education background'


# In[87]:


sns.distplot(df['CoapplicantIncome'],color='blue')


# In[89]:


sns.distplot(df['LoanAmount'],color='blue')


# In[90]:


#categorical features univariate and bivariate analysis


# In[91]:


#categorical - univariate


# In[92]:


df.info()


# In[100]:


df['Loan_Status'].value_counts().plot.bar()


# In[101]:


df['Gender'].value_counts().plot.bar()


# In[102]:


df['Married'].value_counts().plot.bar()


# In[104]:


df['Self_Employed'].value_counts().plot.bar()


# In[105]:


df['Credit_History'].value_counts().plot.bar()


# In[106]:


#ordinal features


# In[111]:


df['Dependents'].value_counts(normalize=True).plot.bar(figsize=(16,6),title='Dependents')


# In[119]:


df['Education'].value_counts(normalize=True).plot.bar(figsize=(8,6),title='education level')


# In[115]:


df['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(16,6),title='property_area')


# In[120]:


#bivariate analysis


# In[121]:


df.info()


# In[133]:


Gender=pd.crosstab(df['Gender'],df['Loan_Status'])
Gender


# In[140]:


Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,figsize=(6,6))


# In[150]:


married=pd.crosstab(df['Married'],df['Loan_Status'])
married.plot.bar(stacked=True)


# In[165]:


married.div(married.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(6,6))


# In[169]:


Education=pd.crosstab(df['Education'],df['Loan_Status'])
Education


# In[176]:


Education.div(Education.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(10,6))


# In[178]:


df.info()


# In[179]:


Dependents=pd.crosstab(df['Dependents'],df['Loan_Status'])
Self_Employed=pd.crosstab(df['Self_Employed'],df['Loan_Status'])


# In[184]:


Dependents.div(Dependents.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(10,6))


# In[185]:


Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(10,6))


# In[186]:


Credit_History=pd.crosstab(df['Credit_History'],df['Loan_Status'])
Property_Area=pd.crosstab(df['Property_Area'],df['Loan_Status'])


# In[188]:


Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(10,6))


# In[189]:


Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(10,6))


# # numerical independent variable vs loan status

# In[200]:


df.groupby('Loan_Status')["ApplicantIncome"].mean().plot.bar()
 


# In[205]:


bins=[0,2500,4000,6000,50000]
group=['low','medium','High','Very High']
df['Income_bin']=pd.cut(df['ApplicantIncome'],bins,labels=group)


# In[209]:


Income_bin=pd.crosstab(df['Income_bin'],df['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True)


# In[232]:


bins1=[0,2500,6000,50000]
group1=['low','medium','High']


# In[233]:


df['coIncome_bin']=pd.cut(df['CoapplicantIncome'],bins1,labels=group1)
coIncome_bin=pd.crosstab(df['coIncome_bin'],df['Loan_Status'])
coIncome_bin.div(coIncome_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(6,6))


# In[226]:


df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']


# In[234]:


bins1=[0,2500,6000,50000]
group1=['low','medium','High']
df['TotalIncome_bin']=pd.cut(df['TotalIncome'],bins1,labels=group1)
TotalIncome_bin=pd.crosstab(df['TotalIncome_bin'],df['Loan_Status'])
TotalIncome_bin.div(TotalIncome_bin.sum(1).astype(float),axis=0).plot.bar(stacked=True,figsize=(6,6))


# In[237]:


df.info()


# In[239]:


df=df.drop(['Income_bin','coIncome_bin','TotalIncome','TotalIncome_bin'],axis=1)


# In[256]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[257]:


df['Loan_Status']=encoder.fit_transform(df['Loan_Status'])


# In[260]:


df['Dependents']=encoder.fit_transform(df['Dependents'])


# In[383]:


matrix=df.corr()
matrix


# In[266]:


sns.heatmap(matrix,vmax=.8,square=True,cmap='BuPu')
#heat map to find correlation


# In[335]:


df['Loan_ID']=encoder.fit_transform(df['Loan_ID'])


# In[336]:


df['Gender']=encoder.fit_transform(df['Gender'])


# In[337]:


df['Married']=encoder.fit_transform(df['Married'])


# In[338]:


df['Education']=encoder.fit_transform(df['Education'])


# In[339]:


df['Self_Employed']=encoder.fit_transform(df['Self_Employed'])


# In[340]:


df['Property_Area']=encoder.fit_transform(df['Property_Area'])


# In[341]:


#to make the outlier treatment we use log transformation as it can make it look like a normal distribution


# In[342]:


df['LoanAmount']=np.log(df['LoanAmount'])


# In[343]:


df['LoanAmount'].hist(bins=20)


# In[344]:


sns.distplot(df['LoanAmount'],color='blue')


# In[345]:


#distribution become exact as normal distribution


# In[346]:


#stratified k fold cross validation


# In[347]:


df1=pd.read_csv('C:/Users/vinodh/Downloads/train_ctrUa4K.csv')


# In[350]:


df1['Loan_Status']=encoder.fit_transform(df1['Loan_Status'])


# In[369]:


y=df1.iloc[:,12].values


# In[372]:


x=df


# In[373]:


from sklearn.model_selection import train_test_split


# In[461]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[462]:


from sklearn.tree import DecisionTreeClassifier


# In[463]:


DTC=DecisionTreeClassifier()


# In[464]:


X_train.isnull().sum()


# In[465]:


df


# In[466]:


df1['LoanAmount'].fillna(df1['LoanAmount'].mean(),inplace=True)


# In[467]:


df1.isnull().sum()


# In[468]:


df['LoanAmount'].fillna(df1['LoanAmount'],inplace=True)


# In[469]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[470]:


DTC.fit(X_train,Y_train)


# In[471]:


y_pred=DTC.predict(X_test)


# In[472]:


# to find accuracy score


# In[473]:


from sklearn.metrics import accuracy_score


# In[474]:


score=accuracy_score(y_pred,Y_test)


# In[475]:


score


# # naivebayes algorithm to predict

# In[476]:


from sklearn.naive_bayes import GaussianNB


# In[477]:


NBclassifier=GaussianNB()


# In[478]:


NBclassifier.fit(X_train,Y_train)


# In[479]:


y_pred=NBclassifier.predict(X_test)


# In[480]:


from sklearn.metrics import accuracy_score


# In[481]:


score=accuracy_score(y_pred,Y_test)


# In[482]:


score


# In[ ]:





# In[ ]:




