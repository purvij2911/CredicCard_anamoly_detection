# # Random Forest with Random Oversampling

# In[103]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 5, 
                           n_jobs=-1,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_over,y_over)


# In[104]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# #### Random Forest with optimal hyperparameters

# In[105]:


# Model with optimal hyperparameters
rf_over_model = grid_search.best_estimator_

rf_over_model.fit(X_over, y_over)


# #### Model evatuation on train data

# In[106]:


# Predictions on the train set
y_train_pred = rf_over_model.predict(X_over)
display_scores(y_over, y_train_pred)


# In[107]:


# Predicted probability
y_train_pred_proba = rf_over_model.predict_proba(X_over)[:,1]
# Plot the ROC curve
draw_roc(y_over, y_train_pred_proba)


# #### Predictions on the test set

# In[108]:


# Evaluating model on the test data
y_test_pred = rf_over_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[109]:


# Predicted probability
y_test_pred_proba = rf_over_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100%
#     - F1 score: 99.58%
# - Test set
#     - ROC score : 97%
#     - F1 score: 69.92%