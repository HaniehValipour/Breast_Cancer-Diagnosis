#!/usr/bin/env python
# coding: utf-8

# # Installing the necessary libraries

# In[1]:


get_ipython().system(' pip install pandas')
get_ipython().system(' pip install matplotlib')
get_ipython().system(' pip install seaborn')


# # Importing the necessary libraries

# In[2]:


#Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the data

# In[3]:


df = pd.read_csv(r"C:\Users\hanie\OneDrive\Desktop\folder\wdbc2.csv")


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


df[["radius_mean","perimeter_mean","area_mean","texture_mean"]].describe()


# In[7]:


df.info()


# In[8]:


df.drop(["id"],axis = 1,inplace = True)


# In[9]:


np.sum(df.isna())


# In[10]:


df.head()


# In[11]:


df["diagnosis"].value_counts()


# In[12]:


sns.countplot(data=df, x='diagnosis')


# In[13]:


sns.barplot(data=df, x='diagnosis', y='area_mean')


# In[14]:


sns.barplot(data=df, x='diagnosis', y='radius_mean')


# In[15]:


plt.figure(figsize=(5,4))
sns.lineplot(x = df["radius_mean"],y=df["perimeter_mean"],hue = df["diagnosis"])


# # Correlation Matrix

# In[16]:


dfg=df.drop(['diagnosis'],axis = 1,inplace = False)

corrMatrix=dfg.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corrMatrix, annot=True,ax=ax)
plt.show()


# In[17]:


sns.swarmplot(data=df, x="diagnosis", y="radius_mean")


# In[18]:


sns.swarmplot(data=df, x="diagnosis", y="area_mean")


# # Scatter Plot

# In[19]:


plt.figure(figsize=(6,4))
sns.scatterplot(x=df["radius_mean"],y= df["texture_mean"],hue = df["diagnosis"])


# In[20]:


sns.kdeplot(df["area_mean"],shade = True)


# In[21]:


X = df.drop(["diagnosis"],axis = 1)

df["diagnosis"] = df["diagnosis"].map({"M":0, "B":1}) 
y = df.diagnosis


# In[22]:


pip install -U scikit-learn


# # Preprocessing the data

# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)


# # Spliting the dataset into Train and Test sets

# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# # Classification

# In[25]:


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score


# In[26]:


models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB']
acc_score=[]

for model in range(len(models)):
    clf=models[model]
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    acc_score.append(accuracy_score(pred,y_test))
     
d={'Classification_Algorithm':model_names,'Accuracy':acc_score}
models_summary_table=pd.DataFrame(d)


# # Comparing different classification machine learning models

# In[27]:


models_summary_table


# In[28]:


sns.barplot(y='Classification_Algorithm',x='Accuracy',data=models_summary_table)
plt.xlabel('Learning Models')
plt.ylabel('Accuracy scores')
plt.title('Accuracy levels of different classification models')

