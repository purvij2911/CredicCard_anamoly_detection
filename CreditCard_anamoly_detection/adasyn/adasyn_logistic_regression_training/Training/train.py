# Importing computational packages
import numpy as np
import pandas as pd
import boto3
import json
import joblib
import pickle
import tempfile
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn import over_sampling
import warnings
warnings.filterwarnings("ignore")


def handler(event,context):
    """
    
    Returns
    -------
    message
        message with sucess of execution
    """    
    try:
        #Step 1: Loading the Data
        X_train, y_train,X_test,y_test= load_dataset()

        #Step 2: Preprocess the Training sample
        X_train = preprocessing(X_train)

        #Step3: Oversampling Technique
        X_train_adasyn,y_train_adasyn=oversample(X_train,y_train)

        #Step 4: Train the Model
        model = training(X_train_adasyn,y_train_adasyn)

        #Step 5: Save the model to s3
        savemodel(model)

        #Step 6: Predictions
        #X_test = preprocessing(X_test)
        #prediction(model,X_test)

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

        

def load_dataset():
        #ref_curated_bucket = event["ref_curated_bucket"]
        #ref_curated_file = event["ref_curated_file"]
        s3 = boto3.client('s3') 
        #obj = s3.get_object(Bucket= ref_curated_bucket, Key= ref_curated_file) 
        obj = s3.get_object(Bucket= 'jupyternotebook-purvi', Key= 'creditcard.csv')
        # read data into pandas dataframe
        df = pd.read_csv(obj['Body'], sep= ',')

        #Splitting Tarin and test
        y= df["Class"]
        X = df.drop("Class", axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42,stratify=y)
        return(X_train, y_train,X_test,y_test)

def preprocessing(X_train):
    
    # As PCA is already performed on the dataset from V1 to V28 features, we are scaling only Amount field
    scaler = RobustScaler()
    # Scaling the train data
    X_train[["Amount"]] = scaler.fit_transform(X_train[["Amount"]])
    #Handling skew features 
    var = X_train.columns
    skew_list = []
    for i in var:
        skew_list.append(X_train[i].skew())

    tmp = pd.concat([pd.DataFrame(var, columns=["Features"]), pd.DataFrame(skew_list, columns=["Skewness"])], axis=1)
    tmp.set_index("Features", inplace=True)

    # Filtering the features which has skewness less than -1 and greater than +1
    skewed = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] <-1 )].index


    # ### There is skewness present in the distribution of the above features:
    # - Power Transformer package present in the <b>preprocessing library provided by sklearn</b> is used to make the distribution more gaussian

    # preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
    pt = PowerTransformer()

    # Fitting the power transformer in train data
    X_train[skewed] = pt.fit_transform(X_train[skewed])
    return(X_train)



def oversample(X_train,y_train):
    
    ada = over_sampling.ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = ada.fit_resample(X_train, y_train)

    return(X_train_adasyn,y_train_adasyn)

def training(X_train,y_train):
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
    model_cv.fit(X_train, y_train)

    #Getting the best model
    logreg_adasyn_model = model_cv.best_estimator_

    # Fitting the model 
    logreg_adasyn_model.fit(X_train, y_train)

    return(logreg_adasyn_model)

def savemodel(model):
    # Set Param of Bucket and Folder
    model_bucket = "wcc-sandbox-dev-operational-code" 
    model_file = "ds-models/fraud-detection/adasyn_logisticregression/adasyn_logistic_reg.sav"
    s3_resource = boto3.resource('s3')
    pickle_byte_obj = pickle.dumps(model)
    s3_resource.Object(model_bucket,model_file).put(Body=pickle_byte_obj)






