#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


from sklearn import datasets
breast_cancer = datasets.load_breast_cancer()
df= pd.read_csv(breast_cancer.filename)
#help(pd.read_csv)
print(breast_cancer.filename)
print(breast_cancer.feature_names)
df.head()


# In[3]:


df_with_filepath=pd.read_csv(r'C:\ProgramData\Anaconda3\lib\site-packages\sklearn\datasets\data\breast_cancer.csv')
df_with_filepath.head()


# In[4]:


df_with_delimiter=pd.read_csv(breast_cancer.filename, delimiter=',')
df_with_delimiter.head()


# In[5]:


df_with_columns=pd.read_csv(r'C:\ProgramData\Anaconda3\lib\site-packages\sklearn\datasets\data\breast_cancer.csv', delimiter=',', header= 1, 
                            names= ['mean radius' ,'mean texture' ,
                                    'mean perimeter', 'mean area','mean smoothness', 
                                    'mean compactness', 'mean concavity','mean concave points' 
                                    ,'mean symmetry', 'mean fractal dimension',
 'radius error' ,'texture error', 'perimeter error', 'area error',
 'smoothness error', 'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius' ,'worst texture' ,'worst perimeter', 'worst area',
 'worst smoothness', 'worst compactness' ,'worst concavity',
 'worst concave points' ,'worst symmetry', 'worst fractal dimension', 'Variety'])
df_with_columns.head()


# In[6]:


print(df_with_columns.info())


# In[7]:


print(df_with_columns.shape)
print(df_with_columns["Variety"].value_counts())
df_with_columns["Variety"].hist()


# In[8]:


print(df_with_columns.shape)
print(df_with_columns.columns)


# In[9]:


print(df_with_columns["Variety"].value_counts())


# In[11]:


print(df_with_columns.describe())


# In[12]:


import seaborn as sns #visualisation
import matplotlib.pyplot as plt
for ojha, feature in enumerate(list(df_with_columns.columns)[:-1]):
    fg = sns.FacetGrid(df_with_columns, hue='Variety', height=5)
    fg.map(sns.distplot, feature).add_legend()
    plt.show()


# In[13]:


sns.boxplot(x='Variety',y='worst radius', data=df_with_columns) 
plt.show()


# In[14]:


sns.violinplot(x='Variety',y='worst radius', data=df_with_columns, size=20)
plt.show()


# In[15]:


df_with_columns.plot(kind='scatter',x='worst radius',y='radius error', c= 'g')
plt.show()


# In[16]:


sns.set_style("whitegrid");
sns.FacetGrid(df_with_columns, hue="Variety", height=10)    .map(plt.scatter, "worst radius", "radius error")    .add_legend();
plt.show();


# In[ ]:





# In[ ]:




