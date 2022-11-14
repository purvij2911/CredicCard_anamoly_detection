# # Decision Tree on balanced data with SMOTE

# In[131]:


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
grid_search.fit(X_train_smote,y_train_smote)


# In[132]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[133]:


print(grid_search.best_estimator_)


# #### Model with optimal hyperparameters

# In[134]:


grid_search.best_params_


# In[135]:


# Model with optimal hyperparameters
dt_smote_model = grid_search.best_estimator_

dt_smote_model.fit(X_train_smote, y_train_smote)


# #### Evaluating the model on train data

# In[136]:


# Predictions on the train set
y_train_pred_smote = dt_smote_model.predict(X_train_smote)
display_scores(y_train_smote, y_train_pred_smote)


# In[137]:


# Predicted probability
y_train_pred_proba = dt_smote_model.predict_proba(X_train_smote)[:,1]
# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[138]:


# Evaluating model on the test data
y_pred = dt_smote_model.predict(X_test)
display_scores(y_test, y_pred)


# In[139]:


# Predicted probability
y_test_pred_smote = dt_smote_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_smote)


# #### Model Summary
# - Train set
#     - ROC score : 100%
#     - F1 score: 98.93%
# - Test set
#     - ROC score : 95%
#     - F1 score: 17.37%