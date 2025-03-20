import pandas as pd

# Load the dataset
data = pd.read_csv('data/raw/heart_disease_uci.csv')

# Drop faulty data
data = data[data['ca'] < 4]

print(f'The length of the data is {len(data)}')

# Rename the columns
data = data.rename(
    columns={
        'cp': 'chest_pain_type',
        'trestbps': 'resting_blood_pressure',
        'chol': 'cholesterol',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'resting_electrocardiogram',
        'thalch': 'max_heart_rate_achieved',
        'exang': 'exercise_induced_angina',
        'oldpeak': 'st_depression',
        'slope': 'st_slope',
        'ca': 'num_major_vessels',
        'thal': 'thalassemia'
    },
    errors="raise"
)

data.drop(columns=['dataset','id'], inplace=True)

print(data.isnull().sum())
# Fill missing values in numerical columns with the median
data["resting_blood_pressure"].fillna(data["resting_blood_pressure"].median(), inplace=True)
data["cholesterol"].fillna(data["cholesterol"].median(), inplace=True)
data["max_heart_rate_achieved"].fillna(data["max_heart_rate_achieved"].median(), inplace=True)
data["st_depression"].fillna(data["st_depression"].median(), inplace=True)

# Fill missing values in categorical columns with the most frequent value
for col in ["fasting_blood_sugar", "resting_electrocardiogram", "exercise_induced_angina", "st_slope", "num_major_vessels", "thalassemia"]:
    data[col].fillna(data[col].mode()[0], inplace=True)
    
print(data.isnull().sum())


# # Convert categorical values to meaningful labels
data.loc[data['sex'] == 'Female', 'sex'] = 0
data.loc[data['sex'] == 'Male', 'sex'] = 1

data.loc[data['chest_pain_type'] == "typical angina", 'chest_pain_type'] = 0
data.loc[data['chest_pain_type'] == "atypical angina", 'chest_pain_type'] = 1
data.loc[data['chest_pain_type'] == "non-anginal", 'chest_pain_type'] = 2
data.loc[data['chest_pain_type'] == "asymptomatic", 'chest_pain_type'] = 3

data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype(int)

data.loc[data['resting_electrocardiogram'] == "normal", 'resting_electrocardiogram'] = 0
data.loc[data['resting_electrocardiogram'] == "st-t abnormality", 'resting_electrocardiogram'] = 1
data.loc[data['resting_electrocardiogram'] == "lv hypertrophy", 'resting_electrocardiogram'] = 2

data['exercise_induced_angina'] = data['exercise_induced_angina'].astype(int)

data.loc[data['st_slope'] == "upsloping", 'st_slope'] = 0
data.loc[data['st_slope'] == "flat", 'st_slope'] = 1
data.loc[data['st_slope'] == "downsloping", 'st_slope'] = 2

data.loc[data['thalassemia'] == "fixed defect", 'thalassemia'] = 1
data.loc[data['thalassemia'] == "normal", 'thalassemia'] = 2
data.loc[data['thalassemia'] == "reversable defect", 'thalassemia'] = 3

# # Print the cleaned dataset
print(data.head())



#Save cleanest dataset

data.to_csv(r"C:\Users\sewasew tadele\Desktop\Heart-Disease-Prediction\data\proccesed\heart_disease_cleaned.csv", index=False)

print("✅ Data preprocessing complete! Cleaned data saved.")