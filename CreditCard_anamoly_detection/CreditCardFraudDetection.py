#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection
# 
# In this project we will predict fraudulent credit card transactions with the help of Machine learning models. 
# 
# In order to complete the project, we are going to follow below high level steps to build and select best model.
# - Read the dataset and perform exploratory data analysis
# - Building different classification models on the unbalanced data
# - Building different models on 3 different balancing technique.
#     - Random Oversampling
#     - SMOTE
#     - ADASYN

# In[1]:


# Importing computational packages
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('float_format', '{:f}'.format)


# In[2]:


# Importing visualization packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


# Importing model building packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer


# In[66]:


from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# In[5]:


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# ## Exploratory data analysis

# In[7]:


df = pd.read_csv('creditcard.csv')
df.head()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


#observe the different feature type present in the data
df.dtypes


# In[11]:


df.info()


# ### Handling Missing Values

# In[12]:


# Checking for the missing value present in each columns
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# We can see that there is no missing value present in the dataframe.

# ### Outliers treatment

# As the whole dataset is transformed with PCA, so assuming that the outliers are already treated. Hence, we are not performing any outliers treatment on the dataframe, though we still see outliers available.

# ### Observe the distribution of our classes

# In[13]:


classes=df['Class'].value_counts()
normal_share=round(classes[0]/df['Class'].count()*100,2)
fraud_share=round(classes[1]/df['Class'].count()*100, 2)
print("Non-Fraudulent : {} %".format(normal_share))
print("    Fraudulent : {} %".format(fraud_share))


# In[14]:


# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
plt.figure(figsize=(20,6))

plt.subplot(1,2,1)
ax=sns.countplot(df["Class"])
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.ylabel("Number of transaction")
plt.xlabel("Class")
plt.title("Credit Card Fraud Class - data unbalance")
plt.grid()
plt.subplot(1,2,2)
fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
df_fraud_percentage = pd.DataFrame(fraud_percentage) 
ax=sns.barplot(x='Class',y='Percentage', data=df_fraud_percentage)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
plt.title('Percentage of fraudulent vs non-fraudulent transcations')

plt.grid()


# **Observation**
# 
# The dataset has very high class imbalance. Only 492 records are there among 284807 records which are labeld as fradudulent transaction.

# In[15]:


# Create a scatter plot to observe the distribution of classes with time
sns.scatterplot( df["Class"],df["Time"],hue=df["Class"])
plt.title("Time vs Class scatter plot")
plt.grid()


# **Observation**
# 
# There is not much insight can be drawn from the distribution of the fraudulent transaction based on time as fraudulent/non-fraudulent both transaction are distributed over time.

# In[16]:


# Create a scatter plot to observe the distribution of classes with Amount
sns.scatterplot(df["Class"],df["Amount"],hue=df["Class"])
plt.title("Amount vs Class scatter plot")
plt.grid()


# **Observation**
# 
# Clearly low amount transactions are more likely to be fraudulent than high amount transaction.

# In[17]:


#plotting the correlation matrix
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize = (10,10))
sns.heatmap(df.corr(),cmap='rainbow')
plt.show()


# There are no features which there is high correlatation , corr > .75

# #### Plotting the distributions of all the features

# In[18]:


# Plotting all the variable in displot to visualise the distribution
var = list(df.columns.values)
# dropping Class columns from the list
var.remove("Class")

i = 0
t0 = df.loc[df['Class'] == 0]
t1 = df.loc[df['Class'] == 1]

plt.figure()
fig, ax = plt.subplots(10,3,figsize=(30,45));

for feature in var:
    i += 1
    plt.subplot(10,3,i)
    sns.kdeplot(t0[feature], bw=0.5,label="0")
    sns.kdeplot(t1[feature], bw=0.5,label="1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid()
plt.show()


# We can see most of the features distributions are overlapping for both the fraud and non-fraud transactions.

# Dropping `Time` column as this feature is not going to help in the model building.
# #### Understanding from Core Banking Business 

# In[19]:


# Drop unnecessary columns
df = df.drop("Time", axis = 1)
df.head()


# ### Splitting the data into train & test data

# In[20]:


y= df["Class"]
X = df.drop("Class", axis = 1)
y.shape,X.shape


# In[21]:


# Spltting the into 80:20 train test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[22]:


# Checking the split of the class label
print(" Fraudulent Count for Full data : ",np.sum(y))
print("Fraudulent Count for Train data : ",np.sum(y_train))
print(" Fraudulent Count for Test data : ",np.sum(y_test))


# ### Feature Scaling using  RobustScaler Scaler

# We need to scale `Amount` column.

# In[23]:


# As PCA is already performed on the dataset from V1 to V28 features, we are scaling only Amount field
scaler = RobustScaler()

# Scaling the train data
X_train[["Amount"]] = scaler.fit_transform(X_train[["Amount"]])

# Transforming the test data
X_test[["Amount"]] = scaler.transform(X_test[["Amount"]])


# In[24]:


X_train.head()


# In[25]:


X_test.head()


# ### Plotting the distribution of a variable to handle skewness

# In[26]:


# plot the histogram of a variable from the dataset to see the skewness
var = X_train.columns

plt.figure(figsize=(30,45))
i=0
for col in var:
    i += 1
    plt.subplot(10,3, i)
    sns.distplot(X_train[col])
    plt.grid()

plt.show()


# Lot of features are highly skewed. So we will check the skewness using skew() and if the skewness is beyond -1 to 1, then we will use power transform to transform the data.

# In[27]:


# Lets check the skewness of the features
var = X_train.columns
skew_list = []
for i in var:
    skew_list.append(X_train[i].skew())

tmp = pd.concat([pd.DataFrame(var, columns=["Features"]), pd.DataFrame(skew_list, columns=["Skewness"])], axis=1)
tmp.set_index("Features", inplace=True)
tmp


# In[28]:


# Filtering the features which has skewness less than -1 and greater than +1
skewed = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] <-1 )].index
skewed.tolist()


# ### There is skewness present in the distribution of the above features:
# - Power Transformer package present in the <b>preprocessing library provided by sklearn</b> is used to make the distribution more gaussian

# In[29]:


# preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
pt = PowerTransformer()

# Fitting the power transformer in train data
X_train[skewed] = pt.fit_transform(X_train[skewed])


# Transforming the test data
X_test[skewed] = pt.transform(X_test[skewed])


# In[30]:


# plot the histogram of a variable from the dataset again to see the result 
var = X_train.columns

plt.figure(figsize=(30,45))
i=0
for col in var:
    i += 1
    plt.subplot(10,3, i)
    sns.distplot(X_train[col])
    plt.grid()

plt.show()


# ## Model Building with imbalanced data
# We are going to build models on below mentioned algorithms and we will compare for the best model. We are not building models on SVM,  and KNN as these algorithms are computationaly expensive and need more computational resources specially for the SVM and KNN. KNN algorithms calculate distance between each data points and then this calculation iterates for all the data points to calcualte the nearest neighbour. This process is computationally very expensive when we have very large data set. We do not have these resource available so we are skipping these models.
#     - Logistic Regression
#     - Decision Tree
#     - RandomForest
#     - XGBoost
# 
# #### Metric selection on imbalance data
# We are going to use ROC-AUC score as the evaluation metric for the model evaluation purpose. As the data is highly imbalanced and we have only 0.17% fraud cases in the whole data, accuracy will not be the right metric to evaluate the model.

# In[31]:


# Class imbalance
y_train.value_counts()/y_train.shape


# # 1. Logistic Regression

# In[32]:


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
model_cv.fit(X_train, y_train)
#print the evaluation result by choosing a evaluation metric
print('Best ROC AUC score: ', model_cv.best_score_)


# In[33]:


#print the optimum value of hyperparameters
print('Best hyperparameters: ', model_cv.best_params_)


# In[34]:


# cross validation results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[35]:


# plot of C versus train and validation scores
plt.figure(figsize=(16, 8))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('ROC AUC')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')
plt.grid()


# #### Logistic Regression with optimal C

# In[36]:


model_cv.best_params_


# In[37]:


# Instantiating the model with best C
log_reg_imb_model = model_cv.best_estimator_

# Fitting the model on train dataset
log_reg_imb_model.fit(X_train, y_train)


# #### Prediction and model evalution on the train set

# In[38]:


# Creating function to display ROC-AUC score, f1 score and classification report
def display_scores(y_test, y_pred):
    '''
    Display ROC-AUC score, f1 score and classification report of a model.
    '''
    print(f"F1 Score: {round(f1_score(y_test, y_pred)*100,2)}%") 
    print("\n\n")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")


# In[39]:


# Predictions on the train set
y_train_pred = log_reg_imb_model.predict(X_train)


# In[40]:


display_scores(y_train, y_train_pred)


# In[41]:


# ROC Curve function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[42]:


# Predicted probability
y_train_pred_proba = log_reg_imb_model.predict_proba(X_train)[:,1]


# In[43]:


# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[44]:


# Making prediction on the test set
y_test_pred = log_reg_imb_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[45]:


# Predicted probability
y_test_pred_proba = log_reg_imb_model.predict_proba(X_test)[:,1]


# In[46]:


# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# We can see very good ROC on the test data set 0.97.

# #### Model Summary
# 
# - Train set
#     -     ROC : 98%
#     - F1 Score: 74.47%
#     
#     
# - Test set
#     -     ROC : 97%
#     - F1 score: 72.83%

# # Decision Tree

# In[47]:


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
grid_search.fit(X_train,y_train)


# In[48]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head()


# In[49]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[50]:


print(grid_search.best_estimator_)


# #### Decision Tree with optimal hyperparameters

# In[51]:


# Model with optimal hyperparameters
dt_imb_model = grid_search.best_estimator_

dt_imb_model.fit(X_train, y_train)


# #### Prediction on the train set

# In[52]:


y_train_pred = dt_imb_model.predict(X_train)
display_scores(y_train, y_train_pred)


# In[53]:


# Predicted probability
y_train_pred_proba = dt_imb_model.predict_proba(X_train)[:,1]

# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[54]:


y_test_pred = dt_imb_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[55]:


# Predicted probability
y_test_pred_proba = dt_imb_model.predict_proba(X_test)[:,1]

# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# 
# - Train set
#     - ROC Score: 94%
#     - F1 score : 68.32%
#     
#     
# - Test set
#     - ROC Score: 94%
#     - F1 score : 61.73%

# # RandomForest

# In[56]:


from sklearn.ensemble import RandomForestClassifier
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
grid_search.fit(X_train,y_train)


# In[57]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head()


# In[58]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[59]:


print(grid_search.best_estimator_)


# #### Random forest with optimal hyperparameters

# In[60]:


# Model with optimal hyperparameters
rf_imb_model = grid_search.best_estimator_

rf_imb_model.fit(X_train, y_train)


# #### Prediction on the train set

# In[61]:


y_train_pred = rf_imb_model.predict(X_train)
display_scores(y_train, y_train_pred)


# In[62]:


# Predicted probability
y_train_pred_proba = rf_imb_model.predict_proba(X_train)[:,1]

# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[63]:


y_test_pred = rf_imb_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[64]:


# Predicted probability
y_test_pred_proba = rf_imb_model.predict_proba(X_test)[:,1]

# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# 
# - Train set
#     - ROC Score: 99%
#     - F1 score : 75.89%
#     
#     
# - Test set
#     - ROC Score: 96%
#     - F1 score : 76.5%

# # XGBoost

# In[67]:


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
model_cv.fit(X_train, y_train)


# In[68]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[69]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", model_cv.best_score_)


# In[70]:


print(model_cv.best_estimator_)


# In[71]:


# Printing best params
model_cv.best_params_


# #### XGBoost model with optimal hyperparameter

# In[72]:


# fit model on training data
xgb_imb_model = model_cv.best_estimator_
xgb_imb_model.fit(X_train, y_train)


# #### Model evaluation on train set

# In[73]:


# Predictions on the train set
y_train_pred = xgb_imb_model.predict(X_train)

display_scores(y_train, y_train_pred)


# In[74]:


# Predicted probability
y_train_pred_proba_imb_xgb = xgb_imb_model.predict_proba(X_train)[:,1]

# Plot the ROC curve
draw_roc(y_train, y_train_pred_proba_imb_xgb)


# #### Evaluating the model on the test set

# In[75]:


# Predictions on the test set
y_test_pred = xgb_imb_model.predict(X_test)
display_scores(y_test, y_test_pred)


# In[76]:


# Predicted probability
y_test_pred_proba = xgb_imb_model.predict_proba(X_test)[:,1]

# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# 
# - Train set
#     - ROC score: 100%
#     - F1 score: 100.0%
# - Test set
#     - ROC score: 97%
#     - F1 score: 86.49%

# **XGBoost model is giving good performance on the unbalanced data among these 3 models. ROC-AUC score on the train data is 100% and on test data 97%.**

# ### Print the important features of the best model to understand the dataset
# - This will not give much explanation on the already transformed dataset
# - But it will help us in understanding if the dataset is not PCA transformed

# In[77]:


var_imp = []
for i in xgb_imb_model.feature_importances_:
    var_imp.append(i)
print('Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-1])+1)
print('2nd Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-2])+1)
print('3rd Top var =', var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-3])+1)


# In[78]:


# Variable on Index-17 and Index-14 seems to be the top 2 variables
top_var_index = var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-1])
second_top_var_index = var_imp.index(np.sort(xgb_imb_model.feature_importances_)[-2])

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]

np.random.shuffle(X_train_0)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [20, 10]

plt.scatter(X_train_1[:, top_var_index], X_train_1[:, second_top_var_index], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], top_var_index], X_train_0[:X_train_1.shape[0], second_top_var_index],
            label='Actual Class-0 Examples')
plt.legend()
plt.grid()


# #### Print the FPR,TPR & select the best threshold from the roc curve for the best model

# In[79]:


print('Train auc =', metrics.roc_auc_score(y_train, y_train_pred_proba_imb_xgb))


# In[80]:


fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_pred_proba_imb_xgb)
threshold = thresholds[np.argmax(tpr-fpr)]
print("Threshold=",threshold)


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


# In[82]:


from collections import Counter
# Befor sampling class distribution
print('Before sampling class distribution:-',Counter(y_train))


# In[83]:


# new class distribution 
print('New class distribution:-',Counter(y_over))


# # Logistic Regrassion with Random Oversampling

# In[84]:


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

# # XGBoost with Random Oversampling

# In[110]:


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
model_cv.fit(X_over, y_over) 


# In[111]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[112]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", model_cv.best_score_)


# In[113]:


print(model_cv.best_estimator_)


# In[114]:


model_cv.best_params_


# #### XGBoost with optimal hyperparameter

# In[115]:


# fit model on training data
xgb_over_model = model_cv.best_estimator_
xgb_over_model.fit(X_over, y_over)


# #### Model evatuation on train data

# In[116]:


# Predictions on the train set
y_train_pred = xgb_over_model.predict(X_over)

display_scores(y_over, y_train_pred)


# In[117]:


# Predicted probability
y_train_pred_proba = xgb_over_model.predict_proba(X_over)[:,1]

# Plot the ROC curve
draw_roc(y_over, y_train_pred_proba)


# #### Model evaluation on the test set

# In[118]:


y_pred = xgb_over_model.predict(X_test)
display_scores(y_test, y_pred)


# In[119]:


# Predicted probability
y_test_pred_proba = xgb_over_model.predict_proba(X_test)[:,1]

# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100.0%
#     - F1 score: 99.99%
# - Test set
#     - ROC score : 97%
#     - F1 score: 83.33%

# ## SMOTE (Synthetic Minority Oversampling Technique)

# ### Print the class distribution after applying SMOTE 

# In[120]:


from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
# Artificial minority samples and corresponding minority labels from SMOTE are appended
# below X_train and y_train respectively
# So to exclusively get the artificial minority samples from SMOTE, we do
X_train_smote_1 = X_train_smote[X_train.shape[0]:]

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]


plt.rcParams['figure.figsize'] = [20, 20]
fig = plt.figure()

plt.subplot(3, 1, 1)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_smote_1.iloc[:X_train_1.shape[0], 0], X_train_smote_1.iloc[:X_train_1.shape[0], 1],
            label='Artificial SMOTE Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], 0], X_train_0[:X_train_1.shape[0], 1], label='Actual Class-0 Examples')
plt.legend()


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

# # Randomforest on balanced data with SMOTE

# In[140]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
model = RandomForestClassifier()

grid_search = GridSearchCV(estimator = model, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 5, 
                           n_jobs=-1,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_smote,y_train_smote)


# In[141]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[142]:


print(grid_search.best_estimator_)


# #### Model with optimal hyperparameters

# In[143]:


grid_search.best_params_


# In[144]:


# Model with optimal hyperparameters
rf_smote_model = grid_search.best_estimator_

rf_smote_model.fit(X_train_smote, y_train_smote)


# #### Evaluating the model on train data

# In[145]:


# Predictions on the train set
y_train_pred_smote = rf_smote_model.predict(X_train_smote)
display_scores(y_train_smote, y_train_pred_smote)


# In[146]:


# Predicted probability
y_train_pred_proba = rf_smote_model.predict_proba(X_train_smote)[:,1]
# Plot the ROC curve
draw_roc(y_train_smote, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[147]:


# Evaluating model on the test data
y_pred = rf_smote_model.predict(X_test)
display_scores(y_test, y_pred)


# In[148]:


# Predicted probability
y_test_pred_smote = rf_smote_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_smote)


# #### Model Summary
# - Train set
#     - ROC score : 100%
#     - F1 score: 98.99%
# - Test set
#     - ROC score : 98%
#     - F1 score: 54.04%

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

# ## ADASYN (Adaptive Synthetic Sampling)

# ### Print the class distribution after applying ADASYN

# In[159]:


import warnings
warnings.filterwarnings("ignore")

from imblearn import over_sampling

ada = over_sampling.ADASYN(random_state=42)
X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)
# Artificial minority samples and corresponding minority labels from ADASYN are appended
# below X_train and y_train respectively
# So to exclusively get the artificial minority samples from ADASYN, we do
X_train_adasyn_1 = X_train_adasyn[X_train.shape[0]:]

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [20, 20]
fig = plt.figure()

plt.subplot(3, 1, 1)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 2)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_adasyn_1.iloc[:X_train_1.shape[0], 0], X_train_adasyn_1.iloc[:X_train_1.shape[0], 1],
            label='Artificial ADASYN Class-1 Examples')
plt.legend()

plt.subplot(3, 1, 3)
plt.scatter(X_train_1[:, 0], X_train_1[:, 1], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], 0], X_train_0[:X_train_1.shape[0], 1], label='Actual Class-0 Examples')
plt.legend()


# # Logistic Regression on balanced data with ADASYN

# In[160]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=42)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as roc-auc
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'roc_auc', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_adasyn, y_train_adasyn)
#print the evaluation result by choosing a evaluation metric
print('Best ROC AUC score: ', model_cv.best_score_)


# In[161]:


#print the optimum value of hyperparameters
print('Best hyperparameters: ', model_cv.best_params_)


# In[162]:


# cross validation results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[163]:


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

# In[164]:


model_cv.best_params_


# In[165]:


# Instantiating the model
logreg_adasyn_model = model_cv.best_estimator_

# Fitting the model 
logreg_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


# #### Evaluating the model with train data

# In[166]:


# Evaluating on test data
y_train_pred = logreg_adasyn_model.predict(X_train_adasyn)
display_scores(y_train_adasyn, y_train_pred)


# In[167]:


# Predicted probability
y_train_pred_proba = logreg_adasyn_model.predict_proba(X_train_adasyn)[:,1]
# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# #### Evaluating on test data

# In[168]:


# Evaluating on test data
y_pred = logreg_adasyn_model.predict(X_test)
display_scores(y_test, y_pred)


# In[169]:


# Predicted probability
y_test_pred_proba = logreg_adasyn_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 97%
#     - F1 score: 90.49%
# - Test set
#     - ROC score : 97%
#     - F1 score: 3.39%

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

# # RandomForest on balanced data with ADASYN

# In[178]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
model = RandomForestClassifier()

grid_search = GridSearchCV(estimator = model, 
                           param_grid = param_grid, 
                           scoring= 'roc_auc',
                           cv = 5, 
                           n_jobs=-1,
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_adasyn,y_train_adasyn)


# In[179]:


# Printing the optimal roc score and hyperparameters
print("Best roc auc score : ", grid_search.best_score_)


# In[180]:


print(grid_search.best_estimator_)


# #### Model with optimal hyperparameters

# In[181]:


# Model with optimal hyperparameters
rf_adasyn_model =grid_search.best_estimator_
rf_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


# #### Evaluating the model on train data

# In[182]:


# Evaluating model on the test data
y_train_pred = rf_adasyn_model.predict(X_train_adasyn)
display_scores(y_train_adasyn, y_train_pred)


# In[183]:


# Predicted probability
y_train_pred_proba = rf_adasyn_model.predict_proba(X_train_adasyn)[:,1]
# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# #### Evaluating the model on the test set

# In[184]:


# Evaluating model on the test data
y_pred = rf_adasyn_model.predict(X_test)
display_scores(y_test, y_pred)


# In[185]:


# Predicted probability
y_test_pred_proba = rf_adasyn_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100%
#     - F1 score: 99.96%
#     
# - Test set
#     - ROC score : 98%
#     - F1 score: 20.63%

# # XGBoost on balanced data with ADASYN

# In[186]:


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
model_cv.fit(X_train_adasyn, y_train_adasyn)


# In[187]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()


# In[188]:


# Printing the optimal score and hyperparameters
print("Best roc auc score : ", model_cv.best_score_)


# In[189]:


print(model_cv.best_estimator_)


# #### Model with optimal hyperparameter

# In[190]:


model_cv.best_params_


# In[191]:


# Model with optimal hyperparameter
xgb_adasyn_model = model_cv.best_estimator_
xgb_adasyn_model.fit(X_train_adasyn,y_train_adasyn)


# #### Evaluating the model on the train data

# In[192]:


# Predicting on the train set
y_train_pred = xgb_adasyn_model.predict(X_train_adasyn)
# Printing the scores
display_scores(y_train_adasyn, y_train_pred)


# In[193]:


# Predicted probability
y_train_pred_proba = xgb_adasyn_model.predict_proba(X_train_adasyn)[:,1]
# Plot the ROC curve
draw_roc(y_train_adasyn, y_train_pred_proba)


# #### Evaluating the model on test data

# In[194]:


y_pred = xgb_adasyn_model.predict(X_test)
display_scores(y_test, y_pred)


# In[195]:


# Predicted probability
y_test_pred_proba = xgb_adasyn_model.predict_proba(X_test)[:,1]
# Plot the ROC curve
draw_roc(y_test, y_test_pred_proba)


# #### Model Summary
# - Train set
#     - ROC score : 100.0%
#     - F1 score: 100.0%
# - Test set
#     - ROC score : 98%
#     - F1 score: 77.63%

# ### Select the oversampling method which shows the best result on a model
# We have used several balancing technique to solve the minority class imbalance. We have used Random Oversampling, SMOTE, and Adasyn technique to balance the dataset and then we performed logistic regression, random forest and XGBoost algorithms to build models on each sampling method.
# 
# After conducting the experiment on each oversampling method, we have found that XGBoost model is performing well on the  dataset which is balanced with AdaSyn technique. We got ROC score 100% on train data and 98% on the test data and F1 score 100% on train data and 78% in the test data. 
# 
# Hence, we conclude that the `XGBoost model with Adasyn` is the best model.

# ### Print the important features of the best model to understand the dataset

# In[196]:


var_imp = []
for i in xgb_adasyn_model.feature_importances_:
    var_imp.append(i)
print('Top var =', var_imp.index(np.sort(xgb_adasyn_model.feature_importances_)[-1])+1)
print('2nd Top var =', var_imp.index(np.sort(xgb_adasyn_model.feature_importances_)[-2])+1)
print('3rd Top var =', var_imp.index(np.sort(xgb_adasyn_model.feature_importances_)[-3])+1)


# In[197]:


# Variable on Index-14 and Index-4 seems to be the top 2 variables
top_var_index = var_imp.index(np.sort(xgb_adasyn_model.feature_importances_)[-1])
second_top_var_index = var_imp.index(np.sort(xgb_adasyn_model.feature_importances_)[-2])

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]

np.random.shuffle(X_train_0)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [20, 10]

plt.scatter(X_train_1[:, top_var_index], X_train_1[:, second_top_var_index], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], top_var_index], X_train_0[:X_train_1.shape[0], second_top_var_index],
            label='Actual Class-0 Examples')
plt.legend()
plt.grid()


# #### Print the FPR,TPR & select the best threshold from the roc curve

# In[198]:


print('Train auc =', metrics.roc_auc_score(y_train_adasyn, y_train_pred_proba))


# In[199]:


fpr, tpr, thresholds = metrics.roc_curve(y_train_adasyn, y_train_pred_proba )
threshold = thresholds[np.argmax(tpr-fpr)]
print(threshold)


# We have found that 74.6% is the threshold for which TPR is the highest and FPR is the lowest and we get 100% ROC score on the train data.

# ### Summary to the business
# Here, we have to focus on a high recall in order to detect actual fraudulent transactions in order to save the banks from high-value fraudulent transactions,
# 
# After performing several models, we have seen that in the balanced dataset with ADASYN technique the XGBoost model has good ROC score(98%) and also high Recall(87%). Hence, we can go with the XGBoost model here.

# In[201]:


#creating the summary table 
output = pd.DataFrame([['IMBALANCED DATASET','','','',''],
                      ['Logistic Reg','74.47%','98%','72.83%','97%'],
                      ['Decision Tree','68.32%','94%','61.73%','94%'],
                      ['Random Forest','75.89%','99%','76.5%','96%'],
                      ['XG Boost','100%','100%','86.49%','97%'],
                      ['RANDOM OVERSAMPLING','','','',''],
                      ['Logistic Reg','94.95%','99%','10.11%','97%'],
                      ['Decision Tree','99.52%','100%','21.2%','91%'],
                      ['Random Forest','99.58%','100%','69.92%','97%'],
                      ['XG Boost','100%','99.99%','83.33%','97%'],
                      ['SMOTE','','','',''],
                      ['Logistic Reg','94.8%','99%','9.67%','97%'],
                      ['Decision Tree','98.93%','100%','17.37%','95%'],
                      ['Random Forest','98.99%','100%','54.04%','98%'],
                      ['XG Boost','99.89%','100%','52.47%','97%'],
                      ['ADASYN','','','',''],
                      ['Logistic Reg','90.49%','97%','3.39%','97%'],
                      ['Decision Tree','98%','99%','7.64%','94%'],
                      ['Random Forest','99.96%','100%','20.63%','98%'],
                      ['XG Boost','100%','100%','77.63%','98%']],
                      columns= ['Model', 'Train F1 score ','train roc','Test F1 score', ' test ROC'])
output

