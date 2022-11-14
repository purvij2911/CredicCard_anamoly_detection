# We can see that the threshold is 0.86, for which the TPR is the highest and FPR is the lowest and we got the best ROC score.

# ## Model building with balancing Classes
# 
# We are going to perform below over sampling approaches for handling data imbalance and we will pick the best approach based on model performance.
# - Random Oversampling
# - SMOTE
# - ADASYN

# ### Random Oversampling

# In[81]:


from imblearn.over_sampling import RandomOverSampler

# define oversampling strategy
oversample = RandomOverSampler(sampling_strategy='minority')
#fit and apply the transform
X_over, y_over = oversample.fit_resample(X_train, y_train)
X_over.shape, y_over.shape