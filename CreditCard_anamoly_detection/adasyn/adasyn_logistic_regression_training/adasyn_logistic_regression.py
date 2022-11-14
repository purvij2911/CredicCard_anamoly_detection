# Importing computational packages
import numpy as np
import pandas as pd
import boto3
import pickle
import json

# Importing visualization packages
# import seaborn as sns

# Importing model building packages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report
from imblearn import over_sampling



import warnings
warnings.filterwarnings("ignore")
def handler(event, context):
        """
        
        Returns
        -------
        message
            message with sucess of execution
        """    
        try:
            ##############################################################################
            ## Read processed data

            #ref_curated_bucket = event["ref_curated_bucket"]
            #ref_curated_file = event["ref_curated_file"]

            s3 = boto3.client('s3') 
            #obj = s3.get_object(Bucket= ref_curated_bucket, Key= ref_curated_file) 
            obj = s3.get_object(Bucket= 'jupyternotebook-purvi', Key= 'creditcard.csv')
            # read data into pandas dataframe
            df = pd.read_csv(obj['Body'], sep= ',')
            y= df["Class"]
            X = df.drop("Class", axis = 1)
            #y.shape,X.shape

            # Spltting the into 80:20 train test size
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,stratify=y)
            #X_train.shape, X_test.shape, y_train.shape, y_test.shape

            # ### Feature Scaling using  RobustScaler Scaler

            # We need to scale `Amount` column.

            # As PCA is already performed on the dataset from V1 to V28 features, we are scaling only Amount field
            scaler = RobustScaler()

            # Scaling the train data
            X_train[["Amount"]] = scaler.fit_transform(X_train[["Amount"]])

            # Transforming the test data
            X_test[["Amount"]] = scaler.transform(X_test[["Amount"]])

            #X_train.head()
            #X_test.head()

            # Lot of features are highly skewed. So we will check the skewness using skew() and if the skewness is beyond -1 to 1, then we will use power transform to transform the data.
            # Lets check the skewness of the features

            var = X_train.columns
            skew_list = []
            for i in var:
                skew_list.append(X_train[i].skew())

            tmp = pd.concat([pd.DataFrame(var, columns=["Features"]), pd.DataFrame(skew_list, columns=["Skewness"])], axis=1)
            tmp.set_index("Features", inplace=True)
            #tmp

            # Filtering the features which has skewness less than -1 and greater than +1
            skewed = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] <-1 )].index
            #skewed.tolist()



            # ### There is skewness present in the distribution of the above features:
            # - Power Transformer package present in the <b>preprocessing library provided by sklearn</b> is used to make the distribution more gaussian

            # preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
            pt = PowerTransformer()

            # Fitting the power transformer in train data
            X_train[skewed] = pt.fit_transform(X_train[skewed])

            # Transforming the test data
            X_test[skewed] = pt.transform(X_test[skewed])

            # plot the histogram of a variable from the dataset again to see the result 
            var = X_train.columns

            # Class imbalance
            #y_train.value_counts()/y_train.shape
            # ################################################################################################################################################

            ada = over_sampling.ADASYN(random_state=42)
            X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)

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
            #print('Best ROC AUC score: ', model_cv.best_score_)


            # In[161]:


            #print the optimum value of hyperparameters
            #print('Best hyperparameters: ', model_cv.best_params_)


            # In[162]:


            # cross validation results
            cv_results = pd.DataFrame(model_cv.cv_results_)
            #cv_results.head()


            # Instantiating the model
            logreg_adasyn_model = model_cv.best_estimator_

            # Fitting the model 
            logreg_adasyn_model.fit(X_train_adasyn, y_train_adasyn)


            # #### Evaluating the model with train data

            # In[166]:


            # Evaluating on test data
            y_train_pred = logreg_adasyn_model.predict(X_train_adasyn)
            #display_scores(y_train_adasyn, y_train_pred)


            # In[167]:


            # Predicted probability
            y_train_pred_proba = logreg_adasyn_model.predict_proba(X_train_adasyn)[:,1]
            # Plot the ROC curve
            #draw_roc(y_train_adasyn, y_train_pred_proba)


            # #### Evaluating on test data

            # In[168]:


            # Evaluating on test data
            y_pred = logreg_adasyn_model.predict(X_test)
            #display_scores(y_test, y_pred)


            # In[169]:


            # Predicted probability
            y_test_pred_proba = logreg_adasyn_model.predict_proba(X_test)[:,1]
            # Plot the ROC curve
            #draw_roc(y_test, y_test_pred_proba)

            # Set Param of Bucket and Folder
            model_bucket = "wcc-sandbox-dev-operational-code" 
            model_file = "ds-models/fraud-detection/adasyn_logisticregression/adasyn_logisticregression_model.pkl"


            # Save model to S3
            s3_resource = boto3.resource('s3')
            pickle_byte_obj = pickle.dumps(y_pred)
            s3_resource.Object(model_bucket,model_file).put(Body=pickle_byte_obj)


            # #### Model Summary
            # - Train set
            #     - ROC score : 97%
            #     - F1 score: 90.49%
            # - Test set
            #     - ROC score : 97%
            #     - F1 score: 3.39%

            return {
                    'statusCode': 200,
                    'body': json.dumps('End of Lambda Adasyn Model Training - Success!')
                }
        #except:
        except Exception as e:
                return {
                        'statusCode': 400,
                        'body': json.dumps('End of Lambda Adasyn Model Training - Bad Request!'),
                        'exception' : "{}".format(e)
                    }