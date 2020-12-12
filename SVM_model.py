
# coding: utf-8

# In[2]:


# Necessary Libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[13]:


scaled_data=pd.read_csv("Normalized_data.csv")
target=pd.read_csv("Target.csv")
target=target.squeeze()


# In[15]:


#Splitting data to test and train 

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(scaled_data,target,test_size=0.3,random_state=0)


# In[17]:


# Hypermeter Optimization 
from sklearn.svm import SVC
C=np.logspace(-5,15,5,base=2)
H=np.logspace(-15,3,5,base=2)
from sklearn.model_selection import GridSearchCV 

parameters={'C':C,'gamma':H}

svm_model_grid=SVC()

grid_model=GridSearchCV(svm_model_grid,parameters,cv=5)
grid_model.fit(X_train,Y_train)

print('the best parameters : {}'.format(grid_model.best_params_))
print('the best performance: {}'.format(grid_model.best_score_))


# In[19]:


# The best svm model 

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

best_svm_model=SVC(C=32,gamma=8,probability=True)
best_svm_model.fit(X_train,Y_train)
predicted=best_svm_model.predict(X_test)
confusion=confusion_matrix(Y_test,predicted)


# In[21]:


print(confusion) # confusion matrix


# In[22]:


from sklearn.metrics import classification_report

class_report=classification_report(Y_test,predicted)


# In[23]:


print(class_report) # classification report 


# In[24]:


from sklearn.metrics import roc_curve
y_probability=best_svm_model.predict_proba(X_test)[:,1]
fpr,tpr,th=roc_curve(Y_test,y_probability)
plt.plot(fpr,tpr) # roc curve


# In[25]:


from sklearn.metrics import roc_auc_score

score_svm=roc_auc_score(Y_test,y_probability)

print(score_svm) # auc score 

