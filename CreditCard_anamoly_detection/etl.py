# Importing computational packages
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option('float_format', '{:f}'.format)

# Importing model building packages
import warnings
warnings.filterwarnings("ignore")


# ## Exploratory data analysis
df = pd.read_csv('creditcard.csv')


# ### Handling Missing Values

# Checking for the missing value present in each columns
total = df.isnull().sum().sort_values(ascending = False)
percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# We can see that there is no missing value present in the dataframe.
# ### Outliers treatment
# As the whole dataset is transformed with PCA, so assuming that the outliers are already treated. Hence, we are not performing any outliers treatment on the dataframe, though we still see outliers available.
# ### Observe the distribution of our classes

classes=df['Class'].value_counts()
normal_share=round(classes[0]/df['Class'].count()*100,2)
fraud_share=round(classes[1]/df['Class'].count()*100, 2)


fraud_percentage = {'Class':['Non-Fraudulent', 'Fraudulent'], 'Percentage':[normal_share, fraud_share]} 
df_fraud_percentage = pd.DataFrame(fraud_percentage) 

# Drop unnecessary columns
df = df.drop("Time", axis = 1)
print(df.info())

df.to_csv("creditcard_processed.csv", sep=";")