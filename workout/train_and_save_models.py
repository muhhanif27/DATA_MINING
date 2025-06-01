import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import pickle
import os

# Define base directory
BASE_DIR = "E:/Praktikum sem 6/DATA_MINING/workout"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load and preprocess the dataset
def load_data():
    csv_path = os.path.join(BASE_DIR, "workout_fitness_tracker_data.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Drop irrelevant columns
    columns_to_drop = ['User ID', 'Water Intake (liters)', 'VO2 Max', 'Body Fat (%)', 
                       'Mood Before Workout', 'Mood After Workout']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Convert necessary columns to numeric
    numeric_columns = ['Age', 'Height (cm)', 'Weight (kg)', 'Calories Burned', 'Heart Rate (bpm)',
                       'Steps Taken', 'Distance (km)', 'Workout Duration (mins)', 'Sleep Hours',
                       'Daily Calories Intake', 'Resting Heart Rate (bpm)']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    df.dropna(inplace=True)
    return df

# Calculate Workout Efficiency
def calculate_workout_efficiency(row, min_calories, max_calories, min_steps, max_steps, min_distance, max_distance):
    def calculate_bmi(height, weight):
        return weight / (height / 100) ** 2

    def workout_intensity_score(workout_type):
        workout_types = {
            "Cardio": 1.0, "Strength": 1.1, "Yoga": 0.9, "HIIT": 1.2, "Cycling": 1.0, "Running": 1.1
        }
        return workout_types.get(workout_type, 1)

    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value) if max_value > min_value else 0.5

    bmi = calculate_bmi(row['Height (cm)'], row['Weight (kg)'])
    bmi_score = 1.0 if 18.5 <= bmi <= 25 else 0.9 if 25 < bmi <= 30 else 0.8

    calories_score = normalize(row['Calories Burned'], min_calories, max_calories)
    steps_score = normalize(row['Steps Taken'], min_steps, max_steps)
    distance_score = normalize(row['Distance (km)'], min_distance, max_distance)

    max_heart_rate = 220 - row['Age']
    heart_rate_score = row['Heart Rate (bpm)'] / max_heart_rate

    intensity_score = workout_intensity_score(row['Workout Type'])
    age_modifier = 1.05 if row['Age'] < 30 else 1 if row['Age'] <= 45 else 0.95
    gender_modifier = 1.05 if row['Gender'] == "Male" else 1

    workout_efficiency = (
        (calories_score * 0.3) + (steps_score * 0.2) + (distance_score * 0.15) +
        (heart_rate_score * 0.1) + (intensity_score * 0.1) + (bmi_score * 0.05) +
        (age_modifier * 0.025) + (gender_modifier * 0.025)
    )
    return workout_efficiency

# Preprocess data
def preprocess_data(df, selected_features):
    df = df.copy()

    # Calculate Workout Efficiency
    min_calories, max_calories = df['Calories Burned'].min(), df['Calories Burned'].max()
    min_steps, max_steps = df['Steps Taken'].min(), df['Steps Taken'].max()
    min_distance, max_distance = df['Distance (km)'].min(), df['Distance (km)'].max()
    df['Workout Efficiency'] = df.apply(
        lambda row: calculate_workout_efficiency(
            row, min_calories, max_calories, min_steps, max_steps, min_distance, max_distance
        ), axis=1
    )

    # Select features
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features]

    # Encode categorical columns
    nominal_columns = ['Gender', 'Workout Type']
    ordinal_columns = {'Workout Intensity': ['Low', 'Medium', 'High']}

    label_encoder = LabelEncoder()
    for column in nominal_columns:
        if column in df.columns:
            df[column] = label_encoder.fit_transform(df[column])

    for column, order in ordinal_columns.items():
        if column in df.columns:
            mapping = {label: i for i, label in enumerate(order)}
            df[column] = df[column].map(mapping)

    # Standardize numerical columns
    numerical_cols = [
        'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)',
        'Age', 'Height (cm)', 'Weight (kg)', 'Sleep Hours', 'Resting Heart Rate (bpm)',
        'Daily Calories Intake', 'Workout Efficiency'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    scaler = StandardScaler()
    if numerical_cols:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Calculate Efficiency Classification
    percentile_33 = df['Workout Efficiency'].quantile(0.33)
    percentile_66 = df['Workout Efficiency'].quantile(0.66)

    def classify_workout_efficiency(efficiency, low_threshold, high_threshold):
        if efficiency < low_threshold:
            return 0  # Low Efficiency
        elif low_threshold <= efficiency <= high_threshold:
            return 1  # Moderate Efficiency
        else:
            return 2  # High Efficiency

    df['Efficiency Classification'] = df['Workout Efficiency'].apply(
        lambda x: classify_workout_efficiency(x, percentile_33, percentile_66)
    )

    X = df.drop(columns=['Efficiency Classification'])
    y = df['Efficiency Classification']
    return X, y, scaler

# Define scenarios
scenarios = {
    "No Feature Selection": [
        'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Workout Type', 'Workout Duration (mins)',
        'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)', 'Workout Intensity',
        'Sleep Hours', 'Daily Calories Intake', 'Resting Heart Rate (bpm)', 'Workout Efficiency'
    ],
    "Chi-square": [
        'Calories Burned', 'Workout Efficiency', 'Steps Taken', 'Distance (km)', 'Heart Rate (bpm)', 
        'Age', 'Workout Type', 'Workout Duration (mins)', 'Weight (kg)', 'Height (cm)', 'Gender'
    ],
    "Sequential Feature Selection (SFS)": [
        'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'Workout Type', 'Workout Duration (mins)', 
        'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)', 'Workout Efficiency'
    ]
}

# Train and save models
def train_and_save_models():
    df = load_data()

    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    for scenario_name, features in scenarios.items():
        print(f"Training models for {scenario_name}...")
        try:
            X, y, scaler = preprocess_data(df, features)

            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Train Decision Tree
            dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            dt_model.fit(X_resampled, y_resampled)
            dt_path = os.path.join(MODELS_DIR, f'dt_{scenario_name.replace(" ", "_").lower()}.pkl')
            with open(dt_path, 'wb') as f:
                pickle.dump(dt_model, f)

            # Train Naive Bayes
            nb_model = GaussianNB()
            nb_model.fit(X_resampled, y_resampled)
            nb_path = os.path.join(MODELS_DIR, f'nb_{scenario_name.replace(" ", "_").lower()}.pkl')
            with open(nb_path, 'wb') as f:
                pickle.dump(nb_model, f)

            # Save scaler
            scaler_path = os.path.join(MODELS_DIR, f'scaler_{scenario_name.replace(" ", "_").lower()}.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            print(f"Models and scaler for {scenario_name} saved successfully.")
        except Exception as e:
            print(f"Error training models for {scenario_name}: {str(e)}")
            raise

if __name__ == "__main__":
    train_and_save_models()