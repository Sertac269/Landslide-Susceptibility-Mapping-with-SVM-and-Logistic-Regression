
# coding: utf-8

# In[1]:


# Necessary Libraries 
import pandas as pd 
import numpy as np 


# In[2]:


# Getting no_landslide points csv file to a dataframe

row_nolandslide=pd.read_csv("No_Landslide.csv",delimiter=";")
no_landslide=pd.DataFrame()
no_landslide['Slope']=row_nolandslide["RASTERVALU"]
no_landslide['Elevation']=row_nolandslide["RASTERVALU_1"]
no_landslide['Corina']=row_nolandslide['RASTERVALU_12']


# In[3]:


# Getting landslide points csv file to a dataframe

row_landsldie=pd.read_csv("Landslide.csv",delimiter=';')
landslide=pd.DataFrame()
landslide['Slope']=row_landsldie['Slope_slide']
landslide['Elevation']=row_landsldie['Elev_slidee']
landslide['Corina']=row_landsldie['Corine_Slide']


# In[4]:


# Adding target values to landslide and non_landslide points

landslide['Target']=pd.DataFrame(np.ones((4493,1)))
no_landslide['Target']=pd.DataFrame(np.zeros((4491,1)))
Final=landslide.append(no_landslide,ignore_index=True)


# In[5]:


# Changing CORİNE numaric values to categorical values 

Final['Corina'][Final['Corina']==112]="Discontinous Urban Fabric"
Final['Corina'][Final['Corina']==211]="Non-Irrigated Arable Land"
Final['Corina'][Final['Corina']==212]="Permanently irrigated land"
Final['Corina'][Final['Corina']==231]="Pastures"
Final['Corina'][Final['Corina']==242]="Complex cultivation patterns"
Final['Corina'][Final['Corina']==243]="Land principally occupied by agriculture"
Final['Corina'][Final['Corina']==311]="Broad-leaved forest"
Final['Corina'][Final['Corina']==321]="Natural grasslands"
Final['Corina'][Final['Corina']==324]="Transitional woodland"
Final['Corina'][Final['Corina']==332]="Bare rocks"
Final['Corina'][Final['Corina']==333]="Sparsely vegetated areas"
Final['Corina'][Final['Corina']==411]="Inland marshes"
Final['Corina'][Final['Corina']==512]="Water bodies"
Final['Corina'][Final['Corina']==213]='Rice Fields'


# In[7]:


# Determining null values and extracting null data
Final.info()
Final=Final.dropna()


# In[9]:


Final.to_csv(path_or_buf="Final_Data.csv",index=False)

