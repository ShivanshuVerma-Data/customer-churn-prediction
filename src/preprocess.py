import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # 🔥 Feature engineering
    df['AvgCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = df['tenure'] // 12
    df['ChargePerTenure'] = df['MonthlyCharges'] * df['tenure']
    df['IsNewCustomer'] = (df['tenure'] < 12).astype(int)
    df['HighSpender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)

    return df


def encode_target(df):
    df = df.copy()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


def encode_features(df):
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def split_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def preprocess_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_target(df)
    df = encode_features(df)

    return split_data(df)