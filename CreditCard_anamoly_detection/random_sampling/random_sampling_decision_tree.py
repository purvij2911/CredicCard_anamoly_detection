# # Decision Tree with Random Oversampling

# In[95]:


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
grid_search.fit(X_over,y_over)


# In[96]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[97]:


print(grid_search.best_estimator_)


# #### Decision Tree with optimal hyperparameters

# In[98]:


# Model with optimal hyperparameters
dt_over_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 42,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)

dt_over_model.fit(X_over, y_over)


# #### Model evatuation on train data

# In[99]:


# Predictions on the train set
y_train_pred = dt_over_model.predict(X_over)
display_scores(y_over, y_train_pred)


# In[100]:


# Predicted probability
y_train_pred_proba = dt_over_model.predict_proba(X_over)[:,1]
# Plot the ROC curve
draw_roc(y_over, y_train_pred_proba)


# #### Predictions on the test set

# In[101]:


# Evaluating model on the test data
y_test_pred = dt_over_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[102]:


# Predicted probability
y_test_pred_proba = dt_over_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100%
#     - F1 score: 99.52%
# - Test set
#     - ROC score : 91%
#     - F1 score: 21.2%