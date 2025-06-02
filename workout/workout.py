import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import uuid
import os
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Workout Efficiency Classification Dashboard", layout="wide")

# Define base directory
BASE_DIR = "E:/Praktikum sem 6/DATA_MINING/workout"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Title and description
st.title("Workout Efficiency Classification Dashboard")
st.markdown("""
Pilih skenario seleksi fitur dan algoritma untuk mengklasifikasi efisiensi latihan.
Gunakan tab Input Manual untuk memasukkan data sesuai fitur yang dipilih, atau tab Input Batch untuk mengunggah file CSV.
Hasil akan ditampilkan dalam tabel (termasuk Workout Efficiency sebelum dan sesudah scaling),
dengan visualisasi distribusi prediksi untuk input batch, dan dapat diunduh.
""")

# Define scenarios and their features
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

# Define categorical values
categorical_values = {
    'Gender': ['Male', 'Other', 'Female'],
    'Workout Type': ['Cycling', 'Cardio', 'HIIT', 'Strength', 'Yoga', 'Running'],
    'Workout Intensity': ['High', 'Medium', 'Low'],
    'Mood Before Workout': ['Tired', 'Happy', 'Neutral', 'Stressed'],
    'Mood After Workout': ['Fatigued', 'Energized', 'Neutral']
}

# Define algorithms
algorithms = ["Decision Tree", "Naive Bayes"]

# Sidebar for selections
st.sidebar.header("Konfigurasi")
selected_scenario = st.sidebar.selectbox("Pilih Skenario Seleksi Fitur", list(scenarios.keys()))
selected_algorithm = st.sidebar.selectbox("Pilih Algoritma", algorithms)

# Display selected features
st.subheader("Fitur yang Dipilih untuk Prediksi")
st.write([f for f in scenarios[selected_scenario] if f != 'Workout Efficiency'])

# Calculate Workout Efficiency
def calculate_workout_efficiency(row, norm_params):
    def calculate_bmi(height, weight):
        return weight / ((height / 100) ** 2)

    def workout_intensity_score(workout_type):
        workout_types = {
            "Cardio": 1.0, "Strength": 1.1, "Yoga": 0.9, "HIIT": 1.2, "Cycling": 1.0, "Running": 1.1
        }
        return workout_types.get(workout_type, 1)

    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value) if max_value > min_value else 0.5

    bmi = calculate_bmi(row['Height (cm)'], row['Weight (kg)'])
    bmi_score = 1.0 if 18.5 <= bmi <= 25 else 0.9 if 25 < bmi <= 30 else 0.8

    calories_score = normalize(row['Calories Burned'], norm_params['min_calories'], norm_params['max_calories'])
    steps_score = normalize(row['Steps Taken'], norm_params['min_steps'], norm_params['max_steps'])
    distance_score = normalize(row['Distance (km)'], norm_params['min_distance'], norm_params['max_distance'])

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

# Preprocess input data
def preprocess_data(df, selected_features, scaler, norm_params):
    df = df.copy()

    # Calculate Workout Efficiency (Raw)
    df['Workout Efficiency (Raw)'] = df.apply(
        lambda row: calculate_workout_efficiency(row, norm_params), axis=1
    )

    # Check for missing features
    required_features = [f for f in selected_features if f != 'Workout Efficiency']
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        st.error(f"Fitur yang hilang: {missing_features}")
        return None, None

    # Select features for prediction
    df_processed = df.copy()
    df_processed['Workout Efficiency'] = df['Workout Efficiency (Raw)']  # Copy for scaling
    available_features = [f for f in selected_features if f in df_processed.columns]
    df_processed = df_processed[available_features]

    # Encode categorical columns
    nominal_columns = ['Gender', 'Workout Type']
    ordinal_columns = {'Workout Intensity': ['Low', 'Medium', 'High']}

    label_encoder = LabelEncoder()
    for column in nominal_columns:
        if column in df_processed.columns:
            df_processed[column] = label_encoder.fit_transform(df_processed[column])

    for column, order in ordinal_columns.items():
        if column in df_processed.columns:
            mapping = {label: i for i, label in enumerate(order)}
            df_processed[column] = df_processed[column].map(mapping)

    # Standardize numerical columns
    numerical_cols = [
        'Workout Duration (mins)', 'Calories Burned', 'Heart Rate (bpm)', 'Steps Taken', 'Distance (km)',
        'Age', 'Height (cm)', 'Weight (kg)', 'Sleep Hours', 'Resting Heart Rate (bpm)',
        'Daily Calories Intake', 'Workout Efficiency'
    ]
    numerical_cols = [col for col in numerical_cols if col in df_processed.columns]
    if numerical_cols:
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])

    # Store scaled Workout Efficiency
    processed_df = df.copy()
    processed_df['Workout Efficiency (Scaled)'] = df_processed['Workout Efficiency']

    return df_processed, processed_df

# Load model, scaler, and normalization parameters
@st.cache_resource
def load_model_and_scaler(scenario, algorithm):
    scenario_key = scenario.replace(" ", "_").lower()
    algorithm_key = 'dt' if algorithm == "Decision Tree" else 'nb'
    model_path = os.path.join(MODELS_DIR, f'{algorithm_key}_{scenario_key}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{scenario_key}.pkl')
    norm_params_path = os.path.join(MODELS_DIR, f'norm_params_{scenario_key}.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan: {model_path}. Jalankan train_and_save_models.py untuk membuat model.")
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
    if not os.path.exists(scaler_path):
        st.error(f"File scaler tidak ditemukan: {scaler_path}. Jalankan train_and_save_models.py untuk membuat scaler.")
        raise FileNotFoundError(f"File scaler tidak ditemukan: {scaler_path}")
    if not os.path.exists(norm_params_path):
        st.error(f"File parameter normalisasi tidak ditemukan: {norm_params_path}. Jalankan train_and_save_models.py untuk membuat parameter.")
        raise FileNotFoundError(f"File parameter normalisasi tidak ditemukan: {norm_params_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(norm_params_path, 'rb') as f:
        norm_params = pickle.load(f)
    return model, scaler, norm_params

# Create tabs for Manual Input and Batch Input
tab1, tab2 = st.tabs(["Input Manual", "Input Batch"])

with tab1:
    st.subheader("Input Manual")
    with st.form(key='manual_input_form'):
        st.markdown("Masukkan data untuk fitur yang dipilih:")
        
        # Create input fields for selected features (excluding Workout Efficiency)
        input_data = {}
        for column in [f for f in scenarios[selected_scenario] if f != 'Workout Efficiency']:
            if column in categorical_values:
                input_data[column] = st.selectbox(f"{column}", categorical_values[column])
            else:
                input_data[column] = st.number_input(f"{column}", min_value=0.0, step=0.1)
        
        submit_button = st.form_submit_button(label='Prediksi')

    if submit_button:
        # Convert manual input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Load model, scaler, and normalization parameters
        try:
            model, scaler, norm_params = load_model_and_scaler(selected_scenario, selected_algorithm)
        except FileNotFoundError:
            st.stop()

        # Preprocess input data
        X_input, processed_df = preprocess_data(input_df, scenarios[selected_scenario], scaler, norm_params)
        
        if X_input is not None:
            # Make prediction
            prediction = model.predict(X_input)
            prediction_label = pd.Series(prediction).map({
                0: 'Low Efficiency',
                1: 'Moderate Efficiency',
                2: 'High Efficiency'
            }).iloc[0]
            
            # Define colors for efficiency levels (text color and background color)
            style_map = {
                'Low Efficiency': {'text_color': 'white', 'background_color': 'red'},
                'Moderate Efficiency': {'text_color': 'white', 'background_color': 'blue'},
                'High Efficiency': {'text_color': 'white', 'background_color': 'green'}
            }
            
            # Get the style for the current prediction
            current_style = style_map.get(prediction_label, {'text_color': 'black', 'background_color': 'white'})
            
            # Add prediction and Workout Efficiency to output
            # No need to add to output_df for display here, but keep if needed for other purposes
            # output_df = input_df.copy()
            # output_df['Workout Efficiency (Raw)'] = processed_df['Workout Efficiency (Raw)']
            # output_df['Workout Efficiency (Scaled)'] = processed_df['Workout Efficiency (Scaled)']
            # output_df['Predicted Efficiency Classification'] = prediction_label
            
            # Display results with colored prediction and background
            st.subheader("Hasil Prediksi (Input Manual)")
            st.write(f"**Workout Efficiency (Raw):** {processed_df['Workout Efficiency (Raw)'].iloc[0]:.2f}")
            st.write(f"**Workout Efficiency (Scaled):** {processed_df['Workout Efficiency (Scaled)'].iloc[0]:.2f}")
            
            # Display the prediction with bold text and colored background
            st.markdown(
                f"<div style='background-color: {current_style['background_color']}; color: {current_style['text_color']}; padding: 10px; border-radius: 5px; font-weight: bold;'>"
                f"Predicted Efficiency Classification: {prediction_label}"
                f"</div>", 
                unsafe_allow_html=True
            )
            
            # Optionally, display the input data in a dataframe above the styled prediction
            st.subheader("Data Input Anda")
            st.dataframe(input_df) # Display the original input data
            
with tab2:
    st.subheader("Input Batch")
    uploaded_file = st.file_uploader("Unggah CSV untuk prediksi", type=["csv"])

    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        input_df.columns = input_df.columns.str.strip()

        # Load model, scaler, and normalization parameters
        try:
            model, scaler, norm_params = load_model_and_scaler(selected_scenario, selected_algorithm)
        except FileNotFoundError:
            st.stop()

        # Preprocess input data
        X_input, processed_df = preprocess_data(input_df, scenarios[selected_scenario], scaler, norm_params)
        
        if X_input is not None:
            # Make predictions
            predictions = model.predict(X_input)
            prediction_labels = pd.Series(predictions).map({
                0: 'Low Efficiency',
                1: 'Moderate Efficiency',
                2: 'High Efficiency'
            })
            
            # Add predictions and Workout Efficiency to output
            output_df = input_df.copy()
            output_df['Workout Efficiency (Raw)'] = processed_df['Workout Efficiency (Raw)']
            output_df['Workout Efficiency (Scaled)'] = processed_df['Workout Efficiency (Scaled)']
            output_df['Predicted Efficiency Classification'] = prediction_labels
            
            # Display results
            st.subheader("Hasil Prediksi (Batch)")
            st.dataframe(output_df)
            
            # Visualization: Bar chart of prediction distribution
            st.subheader("Distribusi Kelas Prediksi")
            class_counts = prediction_labels.value_counts().reset_index()
            class_counts.columns = ['Kelas', 'Jumlah']
            fig = px.bar(class_counts, x='Kelas', y='Jumlah', 
                         color='Kelas', text='Jumlah',
                         color_discrete_map={
                             'Low Efficiency': '#FF6347',
                             'Moderate Efficiency': '#4682B4',
                             'High Efficiency': '#32CD32'
                         },
                         title="Distribusi Kelas Prediksi")
            fig.update_layout(xaxis_title="Kelas Prediksi", yaxis_title="Jumlah Data")
            st.plotly_chart(fig, use_container_width=True)
            
            # Download button
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Prediksi",
                data=csv,
                file_name=f"predictions_{selected_scenario}_{selected_algorithm}.csv",
                mime="text/csv",
                key=str(uuid.uuid4())
            )

# Footer
st.markdown("---")
st.markdown("Dikembangkan dengan Streamlit | Data: Workout Fitness Tracker")