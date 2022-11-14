# # Decision Tree on balanced data with ADASYN

# In[170]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 5, 
                           n_jobs=-1,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_adasyn,y_train_adasyn)


# In[171]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[172]:


print(grid_search.best_estimator_)


# #### Model with optimal hyperparameters

# In[173]:


# Model with optimal hyperparameters
dt_adasyn_model =grid_search.best_estimator_
dt_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


# #### Evaluating the model on train data

# In[174]:


# Evaluating model on the test data
y_train_pred = dt_adasyn_model.predict(X_train_adasyn)
display_scores(y_train_adasyn, y_train_pred)


# In[175]:


# Predicted probability
y_train_pred_proba = dt_adasyn_model.predict_proba(X_train_adasyn)[:,1]
# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[176]:


# Evaluating model on the test data
y_pred = dt_adasyn_model.predict(X_test)
display_scores(y_test, y_pred)


# In[177]:


# Predicted probability
y_test_pred_proba = dt_adasyn_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 99%
#     - F1 score: 98%
#     
# - Test set
#     - ROC score : 94%
#     - F1 score: 7.64%