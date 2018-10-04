
# coding: utf-8

# In[11]:


import pandas as pd


# In[12]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")


# In[13]:


data = pd.read_csv('score_de_jamonosidad.csv',
                  usecols={'v1','v2','v3','score'})
data
# sns.pairplot(data, hue="score")


# In[14]:


y = data.pop('score')


# In[15]:


from sklearn import linear_model


# In[16]:


clf = linear_model.SGDRegressor()
clf.fit(data,y)
# clf.score(data,y)


# In[19]:


objetivo = pd.read_csv('jamones_por_calificar.csv', usecols={'v1','v2','v3'})
objetivo


# In[20]:


objetivo['score'] = clf.predict(objetivo)
print(objetivo)


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


reg = LogisticRegression()


# In[26]:


reg.fit(data,y)
reg.score(data,y)


# In[27]:


objetivo = pd.read_csv('jamones_por_calificar.csv', usecols={'v1','v2','v3'})
objetivo.head()


# In[28]:


objetivo['score'] = reg.predict(objetivo)
print(objetivo)

