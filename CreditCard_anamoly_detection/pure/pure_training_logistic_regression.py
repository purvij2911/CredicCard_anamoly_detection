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