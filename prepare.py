
import acquire as acq
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def drop_blank_charge(df):
    df2 = df.copy()
    df2 = df2[df2['total_charges'] != ' ']
    df2['total_charges'] = df2.total_charges.astype(float)
    return df2

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
    tdf['streaming_tv'].replace('Yes','2', inplace=True)
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
    tdf['online_backup'].replace('Yes', '2', inplace = True)
    df['online_backup_encode'] = tdf.online_backup.astype('int')
    return df

def online_svc_combine(df):
    df['online_security_backup'] = df.online_security_encode + df.online_backup_encode
    return df

def household_combine(df):
    tdf = df.copy()
    tdf.dependents.replace('No','0', inplace=True)
    tdf.dependents.replace('Yes','2', inplace=True)
    tdf.partner.replace('Yes','1', inplace=True)
    tdf.partner.replace('No','0', inplace=True)
    df['household_type_id'] = tdf.partner.astype('int') + tdf.dependents.astype('int')
    return df

def internet_type_id_encode(df):
    tdf = df.copy()
    tdf.internet_service_type_id.replace(3,0, inplace = True)
    df.internet_service_type_id = tdf.internet_service_type_id
    return df

def gender_encode(df):
    tdf = df.copy()
    tdf.gender.replace('Female', '0',inplace = True)
    tdf.gender.replace('Male', '1',inplace = True)
    df['gender_encode'] = tdf.gender.astype('int')
    return df

def paperless_billing_encode(df):
    tdf = df.copy()
    tdf.paperless_billing.replace('No', '0',inplace = True)
    tdf.paperless_billing.replace('Yes', '1',inplace = True)
    df['paperless_billing_encode'] = tdf.paperless_billing.astype('int')
    return df

def tech_support_encode(df):
    tdf = df.copy()
    tdf['tech_support'].replace('No internet service', '0', inplace = True)
    tdf['tech_support'].replace('No', '1', inplace = True)
    tdf['tech_support'].replace('Yes','2', inplace=True)
    df['tech_support_encode'] = tdf.tech_support.astype('int')
    return df

def device_protection_encode(df):
    tdf = df.copy()
    tdf['device_protection'].replace('No internet service', '0', inplace = True)
    tdf['device_protection'].replace('No', '1', inplace = True)
    tdf['device_protection'].replace('Yes','2', inplace=True)
    df['device_protection_encode'] = tdf.device_protection.astype('int')
    return df

def df_value_counts(df):
    for c in df.columns:
        return df[c].value_counts()

def scale_total_charges(df1,df2):

    a = df1.copy()
    b = df2.copy()
    scaler = MinMaxScaler()
    scaler.fit(a[['total_charges']])
    a['total_charges_scaled'] = scaler.transform(a[['total_charges']])
    b['total_charges_scaled'] = scaler.transform(b[['total_charges']])

    return [a,b]

def scale_monthly_charges(df1,df2):
    a = df1.copy()
    b = df2.copy()
    scaler = MinMaxScaler()
    scaler.fit(a[['monthly_charges']])
    a['monthly_charges_scaled'] = scaler.transform(a[['monthly_charges']])
    b['monthly_charges_scaled'] = scaler.transform(b[['monthly_charges']])

    return [a,b]

def scale_split_data(df1,df2):
    a = df1.copy()
    b = df2.copy()
    a,b = scale_monthly_charges(a,b)
    a,b = scale_total_charges(a,b)
    return [a,b]

def prep_telco(df):
    return df.pipe(drop_blank_charge)\
    .pipe(tenure_yearly)\
    .pipe(encode_churn)\
    .pipe(multiple_lines_encode)\
    .pipe(streaming_movies_encode)\
    .pipe(streaming_tv_encode)\
    .pipe(streaming_combine)\
    .pipe(online_security_encode)\
    .pipe(online_backup_encode)\
    .pipe(online_svc_combine)\
    .pipe(household_combine)\
    .pipe(internet_type_id_encode)\
    .pipe(gender_encode)\
    .pipe(paperless_billing_encode)\
    .pipe(tech_support_encode)\
    .pipe(device_protection_encode)
