import joblib
import logging
from sklearn.metrics import (
  
  accuracy_score,
  balanced_accuracy_score,
  f1_score,
  classification_report,
  
)

# set up logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s = %(message)s")


def evaluate_models():
   #load artifacts
    try:
      scaler = joblib.load('models/scaler.pkl')
      feature_names = joblib.load('models/feature_names.pkl')
      X_test = joblib.load('models/X_test.pkl')
      y_test = joblib.load('models/y_test.pkl')
    except FileNotFoundError as e:
      logging.error(f" error loading artifacts: {e}")
    
    #load models
    models = {
      "Logistic_Regression": joblib.load('models/logistic_regression.pkl'),
      "XGBoost": joblib.load('models/xgboost.pkl')
    }
        # Scale numerical features for models that require it
    numerical_features = ["age", "resting_blood_pressure", "cholesterol", 
                          "max_heart_rate_achieved", "st_depression"]
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # Evaluate each model
    for name, model in models.items():
      logging.info(f"\n Evaluating {name} ... ")
      
      X_eval = X_test_scaled if name == "Logistic_Regression" else X_test
      
      # make predictions
      y_pred = model.predict(X_eval)
      
      # calculate metrics
      logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
      logging.info(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
      logging.info(f"F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
      logging.info("Classification Report:")
      logging.info(classification_report(y_test, y_pred))
if __name__ == "__main__":
    evaluate_models()    