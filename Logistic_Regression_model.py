
# coding: utf-8




import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 





scaled_data=pd.read_csv("Normalized_data.csv")
target=pd.read_csv("Target.csv")
target=target.squeeze()





#Splitting data to test and train 

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(scaled_data,target,test_size=0.3,random_state=0)


# In[6]:


from sklearn.linear_model import LogisticRegression
C_logistic=np.logspace(-5,15,20,base=2)
from sklearn.model_selection import GridSearchCV 

parameters_logistic={'C':C_logistic}
logistic_model_grid=LogisticRegression()
grid_model_logistic=GridSearchCV(logistic_model_grid,parameters_logistic,cv=5)

grid_model_logistic.fit(X_train,Y_train)

print('the best parameters : {}'.format(grid_model_logistic.best_params_))
print('the best performance: {}'.format(grid_model_logistic.best_score_)) #best parameters for logistic regression


# In[7]:


# Best model for logistic regression

best_logistic_model=LogisticRegression(C=0.03125)
best_logistic_model.fit(X_train,Y_train)

predicted_logistic=best_logistic_model.predict(X_test)

predicted_logistic_prob=best_logistic_model.predict_proba(X_test)


# In[9]:


from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,classification_report
confusion_logistic=confusion_matrix(Y_test,predicted_logistic) # confusion matrix
print(confusion_logistic)


# In[10]:


classification_report_logistic=classification_report(Y_test,predicted_logistic) #classification report
print(classification_report_logistic)


# In[11]:


fpr,tpr,th=roc_curve(Y_test,predicted_logistic_prob[:,1])

plt.plot(fpr,tpr) # roc curve (roc_curve_logistic.png)


# In[12]:


logistic_auc_score=roc_auc_score(Y_test,predicted_logistic_prob[:,1]) # Auc score of prediction
print(logistic_auc_score)

