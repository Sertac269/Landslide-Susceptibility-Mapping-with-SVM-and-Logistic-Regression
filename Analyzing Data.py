
# coding: utf-8

# In[1]:


# Necessary Libraries 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


Final=pd.read_csv("Final_Data.csv")


# In[5]:


# Final_Data
print(Final)


# In[8]:


# Getting statistical info from data
Info=Final.describe()
print(Info)


# In[9]:


print(Final[Final['Target']==1].describe())


# In[10]:


print(Final[Final['Target']==0].describe())


# In[11]:


# Visulazing Scatter pairplot of data (Pairplot.png)
sns.pairplot(data=Final,hue='Target')


# In[13]:


# Getting information of pearson correlation values

print(Final[Final['Target']==1].corr())


# In[14]:


print(Final[Final['Target']==0].corr())


# In[15]:


# Seperating Final data to data and target

data=Final.drop('Target',axis=1)
target=Final['Target']


# In[16]:


#Converting categorical data to numeric data

data_dummy=pd.get_dummies(data)


# In[17]:


# Normalization of data

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(data_dummy)
scaled_data=scaler.fit_transform(data_dummy)
scaled_data=pd.DataFrame(scaled_data,columns=data_dummy.columns)


# In[18]:


# Histogram plot of numeric data (Histogram.png)

fig,ax=plt.subplots(figsize=(16,8))



ax.hist(scaled_data['Elevation'])
ax.hist(scaled_data['Slope'])


# In[21]:


scaled_data.to_csv("Normalized_data.csv",index=False)
target.to_csv("Target.csv",index=False)


# In[ ]:







