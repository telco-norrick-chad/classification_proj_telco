import pandas as pd

def get_telco_data():
    return pd.read_csv('telco_churn.csv')

def peekatdata(df):
    print("HEAD:")
    print(df.head(5))

    print('\n \n TAIL:' )
    print(df.tail(5))

    print("\n \n SHAPE:")
    print(df.shape)

    print("\n \n DESCRIBE:")
    print(df.describe())

    print("\n \n INFO")
    print(df.info())

    print("\n \n Missing Values:")
    missing_vals = df.columns[df.isnull().any()]
    print(df.isnull().sum())
