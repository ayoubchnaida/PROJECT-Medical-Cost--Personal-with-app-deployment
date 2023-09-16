import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn


# Load the pre-trained model
model = pickle.load(open('insurance.sav', 'rb'))

# Define function to preprocess input data
def preprocess_input(age, sex, bmi, children, smoker, region):
    # Encode categorical features
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0
    region_encoded = {'southwest': 3, 'southeast': 2, 'northwest': 1, 'northeast': 0}[region]

    # Create a DataFrame with the preprocessed data
    data = pd.DataFrame({'age': [age], 'sex': [sex_encoded], 'bmi': [bmi], 'children': [children],
                         'smoker': [smoker_encoded], 'region': [region_encoded]})

    return data

# Streamlit app
st.image('https://th.bing.com/th/id/R.05c5364f8004b141ada01b3983b6f6a7?rik=2cXI8kmamM7IWQ&pid=ImgRaw&r=0')
st.title('Insurance Charges Prediction')
st.info('Input Features')
st.sidebar.title('Feature Selection')

# Input features
age = st.sidebar.number_input('Age')
sex = st.sidebar.radio('Sex', ['female', 'male'])
bmi = st.sidebar.number_input('BMI')
children = st.sidebar.number_input('Children', 0, 5)
smoker = st.sidebar.radio('Smoker', ['yes', 'no'])
region = st.sidebar.radio('Region', ['southwest', 'southeast', 'northwest', 'northeast'])

input_data = preprocess_input(age, sex, bmi, children, smoker, region)
st.write(input_data)

st.title('Predicted Insurance Charges:')
# Make Predictions
if st.button('Calculate Insurance Charges'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Insurance Charges: ${prediction[0]:.2f}')
