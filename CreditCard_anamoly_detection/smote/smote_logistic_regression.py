# ### 1. Logistic Regression on balanced data with SMOTE

# In[121]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        n_jobs=-1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_smote, y_train_smote)
#print the evaluation result by choosing a evaluation metric
print('Best ROC AUC score: ', model_cv.best_score_)


# In[122]:


#print the optimum value of hyperparameters
print('Best hyperparameters: ', model_cv.best_params_)


# In[123]:


# cross validation results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[124]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('ROC AUC')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')
plt.grid()


# #### Logistic Regression with optimal C

# In[125]:


# Printing best params
model_cv.best_params_


# In[126]:


# Instantiating the model
logreg_smote_model = model_cv.best_estimator_

# Fitting the model with balanced data
logreg_smote_model.fit(X_train_smote, y_train_smote)


# #### Evaluating the model on train data

# In[127]:


# Evaluating on train data
y_train_pred = logreg_smote_model.predict(X_train_smote)
display_scores(y_train_smote, y_train_pred)


# In[128]:


# Predicted probability
y_train_pred_proba_smote = logreg_smote_model.predict_proba(X_train_smote)[:,1]
# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba_smote)


# #### Evaluating on test data

# In[129]:


# Evaluating on test data
y_test_pred = logreg_smote_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[130]:


# Predicted probability
y_test_pred_proba_smote = logreg_smote_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba_smote)


# #### Model Summary
# - Train set
#     - ROC score : 99%
#     - F1 score: 94.8%
#     
#     
# - Test set
#     - ROC score : 97%
#     - F1 score: 9.67%