import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_model():
    print("Loading dataset from UCI ML Repo...")
    from ucimlrepo import fetch_ucirepo
    heart_disease = fetch_ucirepo(id=45)
    
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Impute missing values with median for all columns
    X = X.fillna(X.median())
    
    # Binarize target
    y = (y['num'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_preds):.4f}")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print("\n--- Random Forest Evaluation ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

    print("\nSaving model...")
    model_data = {
        'model': rf_model,
        'scaler': scaler,
        'features': X.columns.tolist()
    }
    joblib.dump(model_data, 'model.pkl')
    print("Model saved as model.pkl")

if __name__ == "__main__":
    train_model()
