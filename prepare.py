
import acquire as acq
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def encode_churn(df):
    encoder = LabelEncoder()
    encoder.fit(df.churn)
    return df.assign(churn_encode = encoder.transform(df.churn))

def tenure_yearly(df):
    df[['tenure_yearly']] = df[['tenure']]//12
    return df

def multiple_lines_encode(df):
    tdf = df.copy()
    tdf['multiple_lines'].replace('No phone service', '0', inplace = True)
    tdf['multiple_lines'].replace('No', '1', inplace = True)
    tdf['multiple_lines'].replace('Yes','2', inplace=True)
    tdf['multiple_lines']=tdf.multiple_lines.astype('int')
    return tdf

# def encode_streaming_tv(df):
#     tdf = df.copy()
#     tdf['streaming_tv'].replace('No internet service', '0', inplace = True)
#     tdf['streaming_tv'].replace('No', '1', inplace = True)
#     tdf['streaming_tv'].replace('Yes', '2', inplace = True)
#     df['streaming_encode']=tdf.streaming_tv.astype('int')
#     return df

# def encode_streaming_movies(df):
#     tdf = df.copy()
#     tdf['streaming_movies'].replace('No internet service', '0', inplace = True)
#     tdf['streaming_movies'].replace('No', '1', inplace = True)
#     tdf['streaming_movies'].replace('Yes', '2', inplace = True)
#     df['streaming_movies_encode']=tdf.streaming_tv.astype('int')
#     return df

def prep_telco(df):
    return df.pipe(encode_churn)\
    .pipe(multiple_lines_encode)
    # \
    # .pipe(encode_streaming_tv)
    # # \
    # .pipe(encode_streaming_movies)

            
