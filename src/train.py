import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import load_data, preprocess_data

def train_model(X_train, y_train):
    model = LogisticRegression(C=0.1, penalty='l2', random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

def save_model(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

if __name__ == "__main__":
    df = load_data('../data/dataset.csv')
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
    
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test_scaled, y_test)
    
    save_model(model, '../models/final_model.pkl')
    print("Model saved to ../models/final_model.pkl")
