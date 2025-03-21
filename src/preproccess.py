import pandas as pd
import os

# Set global option
pd.set_option('future.no_silent_downcasting', True)

def load_and_preprocess_data():
    
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

        # Convert categorical values to meaningful labels
        data['sex'] = data['sex'].replace({"Male": 1, "Female": 0})

        data['chest_pain_type'] = data['chest_pain_type'].replace({
        "typical angina": 0, 
        "atypical angina": 1, 
        "non-anginal": 2, 
        "asymptomatic": 3
        })

        data['resting_electrocardiogram'] = data['resting_electrocardiogram'].replace({
            "normal": 0, 
            "st-t abnormality": 1, 
            "lv hypertrophy": 2
        })

        data['st_slope'] = data['st_slope'].replace({
            "upsloping": 0, 
            "flat": 1, 
            "downsloping": 2
        })

        data['thalassemia'] = data['thalassemia'].replace({
            "fixed defect": 1, 
            "normal": 2, 
            "reversable defect": 3
        })
                
                
        # Convert boolean columns to integers
        data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype(int)
        data['exercise_induced_angina'] = data['exercise_induced_angina'].astype(int)


        categorical_columns = ["sex", "chest_pain_type", "resting_electrocardiogram", "st_slope", "thalassemia"]
        # Convert categorical columns to integer type
        for col in categorical_columns:
            data[col] = data[col].astype(int)
            
        # Print the cleaned dataset
        print(data.head())

        #Save cleanest dataset
        save_path = os.path.join("data", "proccesed", "heart_disease_cleaned.csv")
        data.to_csv(save_path, index=False)
        print("✅ Data preprocessing complete! Cleaned data saved.")
        
        return data