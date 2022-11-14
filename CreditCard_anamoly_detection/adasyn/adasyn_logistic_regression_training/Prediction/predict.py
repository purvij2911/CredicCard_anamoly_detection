import pandas as pd
import numpy as np
import joblib
import json
import pickle
import boto3
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression



def handler(event, context):

    
    # Get parameter for result file
    ssm = boto3.client('ssm')
    model_bucket = "wcc-sandbox-dev-operational-code" 
    model_file = "ds-models/fraud-detection/adasyn_logisticregression/adasyn_logistic_reg.sav"

    try:

        #Step 1: get the model
        model= getmodel()

        #step 2: get the data
        X_test= getdata()
        
        #Step3: Preprocessing
        X_test= preprocess(X_test)

        #Step 4:Prediction
        prediction(model,X_test)


    except:
        return {
            "statusCode": 400,
            "body": json.dumps("Error Lambda Predict Model - Bad Request!")
            }
    
    

def getmodel():
    model_bucket = "wcc-sandbox-dev-operational-code" 
    model_file = "ds-models/fraud-detection/adasyn_logisticregression/adasyn_logistic_reg.sav"
    s3 = boto3.resource('s3')
    model = pickle.loads(s3.Bucket(model_bucket).Object(model_file).get()['Body'].read())
    return(model)

def getdata():
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
    return(X_test)

def preprocess(X_test):
    
    # As PCA is already performed on the dataset from V1 to V28 features, we are scaling only Amount field
    scaler = RobustScaler()
    # Scaling the train data
    X_test[["Amount"]] = scaler.fit_transform(X_test[["Amount"]])
    #Handling skew features 
    var = X_test.columns
    skew_list = []
    for i in var:
        skew_list.append(X_test[i].skew())

    tmp = pd.concat([pd.DataFrame(var, columns=["Features"]), pd.DataFrame(skew_list, columns=["Skewness"])], axis=1)
    tmp.set_index("Features", inplace=True)

    # Filtering the features which has skewness less than -1 and greater than +1
    skewed = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] <-1 )].index


    # ### There is skewness present in the distribution of the above features:
    # - Power Transformer package present in the <b>preprocessing library provided by sklearn</b> is used to make the distribution more gaussian

    # preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
    pt = PowerTransformer()

    # Fitting the power transformer in train data
    X_test[skewed] = pt.fit_transform(X_test[skewed])
    return(X_test)


def prediction(model,X_test):
    y_pred = model.predict(X_test)
    probability=model.predict_proba(X_test)[:,1]
    X_test['prediction']=y_pred
    X_test['probability']=probability


    model_bucket = "wcc-sandbox-dev-operational-code" 
    model_file = "ds-models/fraud-detection/adasyn_logisticregression/prediction.csv"
    s3 = boto3.resource('s3')
    #put the model in temp file
    with tempfile.TemporaryFile() as fp:
        X_test.to_csv(fp)
        joblib.dump(X_test, fp)
        fp.seek(0)
        s3.Bucket(model_bucket).put_object(Body=fp.read(), Key=model_file)



