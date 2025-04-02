import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import accuracy_score, classification_report

# Set up the logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s = %(message)s")

def train_models():
    # Load processed data
    data = pd.read_csv('data/proccesed/heart_disease_cleaned.csv')

    # Split the data to feature and target
    X = data.drop(columns=["num"], axis=1)
    y = data["num"].apply(lambda x: 0 if x == 0 else 1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    numerical_features = [
        "age", "resting_blood_pressure", "cholesterol", 
        "max_heart_rate_achieved", "st_depression"
    ]

    # Apply StandardScaler to only numerical features
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000),
            "params": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "saga"]
            },
            "scaled": True
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='mlogloss'),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2],
                "min_child_weight": [1, 2, 3]
            },
            "scaled": False
        },
    }

    trained_model = {}
    for name, config in models.items():
        logging.info(f"Training {name}...")

        # Select dataset based on scaling flag
        X_tr = X_train_scaled if config.get("scaled", False) else X_train
        X_te = X_test_scaled if config.get("scaled", False) else X_test
        
        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_tr, y_train)
        y_pred = grid_search.predict(X_te)
        
        trained_model[name] = grid_search.best_estimator_
        
        # Training-time evaluation
        logging.info(f"📌 {name} Training Performance:")
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logging.info(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
        logging.info(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
        logging.info(classification_report(y_test, y_pred))

    # Save artifacts for evaluation
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X_test, 'models/X_test.pkl')
    joblib.dump(y_test, 'models/y_test.pkl')
    joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")

    for name, model in trained_model.items():
        joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

    logging.info("All models and artifacts have been saved successfully!")

if __name__ == "__main__":
    train_models()