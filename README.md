# 💓 Heart Disease Prediction System 
-
This project uses machine learning to predict whether a person has heart disease based on medical parameters. The goal is to provide a fast, accessible, and accurate prediction system that can assist in early diagnosis.
---

## 🚀 Features

- 🧠 Predicts heart disease using key health metrics
- 🔢 Uses Diffrent ML models like Logistic Regression and XGBoost
- ⚙️ Input scaling with `StandardScaler`
- 📁 Well-structured for experimentation and future improvements

---

## 🧠 Machine Learning Models Used

- ✅ Logistic Regression
- ✅ XGBoost

The default model is **Logistic Regression**, but others can be used by switching the `model_name` in `predict.py`.

---

## 📁 Project Structure

Heart-Disease-Prediction/ │ ├── models/ # Saved ML models and scaler │ ├── logistic_regression.pkl │ ├── scaler.pkl │ └── feature_names.pkl │ ├── src/ │ ├── train.py # Training and saving models │ └── predict.py # Load model & make predictions │└── evaluatee.py ├── data/ # Raw or cleaned datasets │ ├── requirements.txt # List of Python dependencies ├── README.md # You are here! └── .gitignore

## 📊 Input Features

The model expects **13 features** in the following order:

1. `age`
2. `sex`
3. `chest_pain_type`
4. `resting_blood_pressure`
5. `cholesterol`
6. `fasting_blood_sugar`
7. `resting_electrocardiogram`
8. `max_heart_rate_achieved`
9. `exercise_induced_angina`
10. `st_depression`
11. `st_slope`
12. `num_major_vessels`
13. `thalassemia`

Example:
``` python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
```
- ✅ Example Output
The Person has Heart Disease OR
The Person does not have a Heart Disease


##  Dataset Source
This project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.

## 🔮 Future Improvements

-🌐 Web interface using Flask or React

-📊 Dashboard for model metrics

-☁️ Cloud deployment with API support

-🔄 Data pipelines and better preprocessing

##👤 Author
Sewasew Tadele
Student | ML Enthusiast
GitHub


