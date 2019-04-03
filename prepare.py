
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
    df['phone_id'] = tdf.multiple_lines.astype('int')
    return df

def streaming_movies_encode(df):
    tdf = df.copy()
    tdf['streaming_movies'].replace('No internet service', '0', inplace = True)
    tdf['streaming_movies'].replace('No', '0', inplace = True)
    tdf['streaming_movies'].replace('Yes','1', inplace=True)
    df['movies_encode'] = tdf.streaming_movies.astype('int')
    return df

def streaming_tv_encode(df):
    tdf = df.copy()
    tdf['streaming_tv'].replace('No internet service', '0', inplace = True)
    tdf['streaming_tv'].replace('No', '0', inplace = True)
    tdf['streaming_tv'].replace('Yes','1', inplace=True)
    df['tv_encode'] = tdf.streaming_tv.astype('int')
    return df

def streaming_combine(df):
    df['streaming_services'] = df.tv_encode + df.movies_encode
    return df

def online_security_encode(df):
    tdf = df.copy()
    tdf['online_security'].replace('No internet service', '0', inplace = True)
    tdf['online_security'].replace('No', '0', inplace = True)
    tdf['online_security'].replace('Yes', '1', inplace = True)
    df['online_security_encode'] = tdf.online_security.astype('int')
    return df

def online_backup_encode(df):
    tdf = df.copy()
    tdf['online_backup'].replace('No internet service', '0', inplace = True)
    tdf['online_backup'].replace('No', '0', inplace = True)
    tdf['online_backup'].replace('Yes', '1', inplace = True)
    df['online_backup_encode'] = tdf.online_backup.astype('int')
    return df

def online_svc_combine(df):
    df['online_security_backup'] = df.online_security_encode + df.online_backup_encode
    return df


def household_combine(df):
    tdf = df.copy()
    tdf.dependents.replace('No','0', inplace=True)
    tdf.dependents.replace('Yes','1', inplace=True)
    tdf.partner.replace('Yes','2', inplace=True)
    tdf.partner.replace('No','0', inplace=True)

    df['household_type_id'] = tdf.partner.astype('int') + tdf.dependents.astype('int')
    return df

def prep_telco(df):
    return df.pipe(encode_churn)\
    .pipe(multiple_lines_encode)\
    .pipe(streaming_movies_encode)\
    .pipe(streaming_tv_encode)\
    .pipe(streaming_combine)\
    .pipe(online_security_encode)\
    .pipe(online_backup_encode)\
    .pipe(online_svc_combine)\
    .pipe(household_combine)
