# Importing computational packages
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('float_format', '{:f}'.format)

# Importing model building packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# ## Read processed data
df = pd.read_csv('creditcard_processed.csv', sep=";")
print(df.info())

y= df["Class"]
X = df.drop("Class", axis = 1)
y.shape,X.shape



# Spltting the into 80:20 train test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape



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