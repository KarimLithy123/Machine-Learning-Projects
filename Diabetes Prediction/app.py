import pickle
import streamlit as st

# Load the saved model
saved_model = pickle.load(open('diabeties_model.sav', 'rb'))

# Set up the Streamlit app title and input columns
st.title("Diabetes Disease Prediction")
col1, col2, col3, col4 = st.columns(4)

# Input fields for user data
with col1:
    Gender = st.selectbox('Gender', options=['Male', 'Female'])
    Age = st.number_input('Age', min_value=2, max_value=100)

with col2:
    hypertension = st.selectbox('Hypertension', options=['Yes', 'No'])
    heart_disease = st.selectbox('Heart Disease', options=['Yes', 'No'])

with col3:
    smoking = st.selectbox('Smoking Status', options=['Unknown', 'Current Smoker', 'Ever Smoked', 'Former Smoker', 'Non-Smoker', 'Ex-Smoker'])
    bmi = st.number_input('BMI', min_value=5.0, max_value=120.0)

with col4:
    HbA1c_level = st.number_input('HbA1c Level', min_value=3.0, max_value=9.0)
    glucose_level = st.number_input('Glucose Level', min_value=70, max_value=300)

# Perform prediction when button is clicked
Diabetes_status = ''
if st.button('Diabetes Predict Status'):
    # Convert categorical inputs to numerical values
    gender_mapping = {'Male': 1, 'Female': 0}
    hypertension_mapping = {'Yes': 1, 'No': 0}
    heart_disease_mapping = {'Yes': 1, 'No': 0}
    smoking_mapping = {'Unknown': 0, 'Current Smoker': 1, 'Ever Smoked': 2, 'Former Smoker': 5, 'Non-Smoker': 4, 'Ex-Smoker': 3}

    # Map user inputs to numerical values
    Gender = gender_mapping[Gender]
    hypertension = hypertension_mapping[hypertension]
    heart_disease = heart_disease_mapping[heart_disease]
    smoking = smoking_mapping[smoking]

    # Create input feature array
    input_features = [Gender, Age, hypertension, heart_disease, smoking, bmi, HbA1c_level, glucose_level]
    input_features = [input_features]

    # Perform prediction
    prediction = saved_model.predict(input_features)

    # Set prediction status message
    if prediction[0] == 1:
        Diabetes_status = 'This person has Diabetes.'
    else:
        Diabetes_status = "This person is healthy and doesn't have the disease."

# Display prediction status
st.success(Diabetes_status)
