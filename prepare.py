
import acquire as acq
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def encode_churn(df):
    encoder = LabelEncoder()
    encoder.fit(df.churn)
    return df.assign(churn_encode = encoder.transform(df.churn))

# def encode_streaming_tv(df):
#     encoder = LabelEncoder()
#     encoder.fit(df.streaming_tv)
#     return df.assign(streaming_tv_encode = encoder.transform(df.streaming_tv))

# def encode_streaming_movies(df):
#     encoder = LabelEncoder()
#     encoder.fit(df.streaming_movies)
#     return df.assign(streaming_movies_encode = encoder.transform(df.streaming_movies))

# def encode_online_security(df):
#     encoder = LabelEncoder()
#     encoder.fit(df.online_security)
#     return df.assign(online_security_encode = encoder.transform(df.online_security))

# def encode_online_backup(df):
#     encoder = LabelEncoder()
#     encoder.fit(df.online_backup)
#     return df.assign(online_backup_encode = encoder.transform(df.online_backup))

def tenure_yearly(df):
    df[['tenure_yearly']] = df[['tenure']]//12
    return df    
    return df

def multiple_lines_encode(df):
    tdf = df.copy()
    tdf['multiple_lines'].replace('No phone service', '0', inplace = True)
    tdf['multiple_lines'].replace('No', '1', inplace = True)
    tdf['multiple_lines'].replace('Yes','2', inplace=True)
    tdf['multiple_lines']=tdf.multiple_lines.astype('int')
    return tdf 

def prep_telco(df):
    return df.pipe(encode_churn)
