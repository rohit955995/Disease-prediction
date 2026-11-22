import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
    df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
    df.drop_duplicates(inplace=True)
    return df

def preprocess_data(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_scaler(scaler, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)

if __name__ == "__main__":
    df = load_data('../data/dataset.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    save_scaler(scaler, '../models/scaler.pkl')
    print("Preprocessing completed and scaler saved.")
