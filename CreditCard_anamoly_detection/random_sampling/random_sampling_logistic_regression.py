# Logistic Regression parameters for K-fold cross vaidation
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}
folds = KFold(n_splits=5, shuffle=True, random_state=42)


#perform cross validation
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        n_jobs=-1,
                        verbose = 1,
                        return_train_score=True) 
#perform hyperparameter tuning
model_cv.fit(X_over, y_over)
#print the evaluation result by choosing a evaluation metric
print('Best ROC AUC score: ', model_cv.best_score_)


# In[85]:


#print the optimum value of hyperparameters
print('Best hyperparameters: ', model_cv.best_params_)


# In[86]:


# cross validation results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[87]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('ROC AUC')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')
plt.grid()


# #### Logistic Regression with hyperparameter tuning

# In[88]:


model_cv.best_params_


# In[89]:


# Instantiating the model
logreg_over = LogisticRegression(C=1000)

# Fitting the model with train data
logreg_over_model = logreg_over.fit(X_over, y_over)


# #### Evaluating the model on train data

# In[90]:


# Predictions on the train set
y_train_pred = logreg_over_model.predict(X_over)


# In[91]:


# Printing scores
display_scores(y_over, y_train_pred)


# In[92]:


# Predicted probability
y_train_pred_proba = logreg_over_model.predict_proba(X_over)[:,1]
# Plot the ROC curve
draw_roc(y_over, y_train_pred_proba)


# #### Evaluating on test data

# In[93]:


# Evaluating on test data
y_test_pred = logreg_over_model.predict(X_test)

# Printing the scores
display_scores(y_test, y_test_pred)


# In[94]:


# Predicted probability
y_test_pred_proba = logreg_over_model.predict_proba(X_test)[:,1]

# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 99%
#     - F1 score: 94.95%
# - Test set
#     - ROC score : 97%
#     - F1 score: 10.11%