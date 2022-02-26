#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install prince


# In[3]:


import prince
import pandas as pd
from sklearn import datasets


# In[7]:


X, y = datasets.load_iris(return_X_y=True)
X = pd.DataFrame(data=X, columns=['Sepal length', 'Sepal width', 'Petal length', 'Petal width'])
y = pd.Series(y).map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
X.head()


# In[19]:


pca = prince.PCA(
    n_components=2,#number of components that are computed
    n_iter=3,#number of iterations used for computing the SVD
    rescale_with_mean=True,#substract each column's mean
    rescale_with_std=True,#divide each column by it's standard deviation
    copy=True,# if False then the computations will be done inplace which can have possible side-effects on the input data
    check_input=True,
    engine='auto', # what SVD engine to use (should be one of ['auto', 'fbpca', 'sklearn'])
    random_state=42#controls the randomness of the SVD results.
)
pca = pca.fit(X)


# In[9]:


pca.transform(X).head()


# In[25]:


ax = pca.plot_row_coordinates(
    X,
    ax=None,
    figsize=(8, 8),
    x_component=0, 
    y_component=1,
    labels=None,
    color_labels=y,
    ellipse_outline=False,
    ellipse_fill=True,
    show_points=True
)


# In[12]:


pca.explained_inertia_ 
#The explained inertia represents the percentage of the inertia each principal component contributes


# In[ ]:


#The explained inertia is obtained by dividing the eigenvalues obtained with the SVD by the total inertia, 
#both of which are also accessible.


# In[13]:


pca.eigenvalues_


# In[14]:


pca.total_inertia_


# In[15]:


pca.explained_inertia_


# In[16]:


pca.column_correlations(X)#correlations between the original variables and the principal components.


# In[17]:


pca.row_contributions(X).head()
#each observation contributes to each principal component


# In[22]:


B=pca.inverse_transform(pca.transform(X)).head()
# transform row projections back into their original space 


# In[23]:


print(B)


# In[ ]:




