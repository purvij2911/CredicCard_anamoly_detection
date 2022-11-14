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
#from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import f1_score, classification_report


import warnings
warnings.filterwarnings("ignore")


# ## ADASYN (Adaptive Synthetic Sampling)

# ### Print the class distribution after applying ADASYN

import warnings
warnings.filterwarnings("ignore")

from imblearn import over_sampling

def handler(event, context):
    """
    
    Returns
    -------
    message
        message with sucess of execution
    """    
    try:
        ####################################################################################################################################################################################
        # ## Read processed data

        ref_curated_bucket = event["ref_curated_bucket"]
        ref_curated_file = event["ref_curated_file"]

        s3 = boto3.client('s3') 
        #obj = s3.get_object(Bucket= ref_curated_bucket, Key= ref_curated_file) 
        obj = s3.get_object(Bucket= 'jupyternotebook-purvi', Key= 'creditcard.csv')
        # read data into pandas dataframe
        df = pd.read_csv(obj['Body'], sep= ',')

        # df = pd.read_csv('creditcard_processed.csv', sep=";")

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
        skewed.tolist()



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

        # # XGBoost on balanced data with ADASYN

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


        # cv results
        #cv_results = pd.DataFrame(model_cv.cv_results_)
        #cv_results.head()

        # #### Model with optimal hyperparameter
        #model_cv.best_params_

        # Model with optimal hyperparameter
        xgb_adasyn_model = model_cv.best_estimator_
        xgb_adasyn_model.fit(X_train_adasyn,y_train_adasyn)


        # #### Evaluating the model on the train data

        # Predicting on the train set
        y_train_pred = xgb_adasyn_model.predict(X_train_adasyn)
        # Printing the scores
        #display_scores(y_train_adasyn, y_train_pred)

        # Predicted probability
        y_train_pred_proba = xgb_adasyn_model.predict_proba(X_train_adasyn)[:,1]
        # Plot the ROC curve
        #draw_roc(y_train_adasyn, y_train_pred_proba)


        # #### Evaluating the model on test data

        y_pred = xgb_adasyn_model.predict(X_test)
        #display_scores(y_test, y_pred)


        # Predicted probability
        y_test_pred_proba = xgb_adasyn_model.predict_proba(X_test)[:,1]
        # Plot the ROC curve
        #draw_roc(y_test, y_test_pred_proba)


        # Set Param of Bucket and Folder
        model_bucket = "wcc-sandbox-dev-operational-code" 
        model_file = "ds-models/fraud-detection/adasyn_xgboosting/adasyn_xgboosting_model.pkl"


        # Save model to S3
        s3_resource = boto3.resource('s3')
        pickle_byte_obj = pickle.dumps(y_pred)
        s3_resource.Object(model_bucket,model_file).put(Body=pickle_byte_obj)


        # # #### Model Summary
        # # - Train set
        # #     - ROC score : 100.0%
        # #     - F1 score: 100.0%
        # # - Test set
        # #     - ROC score : 98%
        # #     - F1 score: 77.63%

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