# # XGBoost on balanced data with SMOTE

# In[149]:


# creating a KFold object 
folds = 5

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          


# specify model
xgb_model = XGBClassifier(max_depth=2, n_estimators=200)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        n_jobs=-1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_smote, y_train_smote)


# In[150]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[151]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", model_cv.best_score_)


# In[152]:


print(model_cv.best_estimator_)


# #### Model with optimal hyperparameter

# In[153]:


model_cv.best_params_


# In[154]:


# fit model on training data
xgb_smote_model = model_cv.best_estimator_
xgb_smote_model.fit(X_train_smote, y_train_smote)


# #### Evaluating the model on the train data

# In[155]:


y_train_pred = xgb_smote_model.predict(X_train_smote)
display_scores(y_train_smote, y_train_pred)


# In[156]:


# Predicted probability
y_train_pred_proba = xgb_smote_model.predict_proba(X_train_smote)[:,1]
# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba)


# #### Evaluating the model on test data

# In[157]:


y_pred = xgb_smote_model.predict(X_test)
display_scores(y_test, y_pred)


# In[158]:


# Predicted probability
y_test_pred_proba = xgb_smote_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100.0%
#     - F1 score: 99.89%
#     
# - Test set
#     - ROC score : 97%
#     - F1 score: 52.47%