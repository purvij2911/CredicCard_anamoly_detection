import pandas as pd
import numpy as np
from sklearn.preprocessing import  RobustScaler
from sklearn.preprocessing import PowerTransformer

def preprocess(X_train):
    
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
