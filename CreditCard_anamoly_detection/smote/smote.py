# ## SMOTE (Synthetic Minority Oversampling Technique)

# ### Print the class distribution after applying SMOTE 

# In[120]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)