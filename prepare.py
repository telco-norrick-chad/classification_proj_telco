
import acquire as acq
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def encode_churn(df):
    encoder = LabelEncoder()
    encoder.fit(df.churn)
    return df.assign(churn_encode = encoder.transform(df.churn))
    
def prep_telco(df):
    return df.pipe(encode_churn)

