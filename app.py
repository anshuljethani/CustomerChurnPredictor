import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')



with open('labelencoderr.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)



with open('onehotencoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App Configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

# App Title and Description
st.title('📉 Customer Churn Prediction')
st.markdown("""
    **Predict customer churn based on various factors.**  
    Adjust the inputs on the left and see the prediction on the right.
    """)

# Sidebar for User Input
st.sidebar.header('Input Customer Details')
st.sidebar.markdown('Fill in the details below:')

# Collecting User Input in the Sidebar
geography = st.sidebar.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('👤 Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('📅 Age', 18, 92, value=30)
balance = st.sidebar.number_input('💰 Balance', min_value=0.0, step=100.0)
credit_score = st.sidebar.number_input('💳 Credit Score', min_value=300, max_value=850, step=10)
estimated_salary = st.sidebar.number_input('💼 Estimated Salary', min_value=0.0, step=1000.0)
tenure = st.sidebar.slider('🕒 Tenure (Years)', 0, 10, value=5)
num_of_products = st.sidebar.slider('🛒 Number of Products', 1, 4, value=1)
has_cr_card = st.sidebar.selectbox('💳 Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('🔄 Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction Section
st.subheader('Prediction Results')
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the Churn Probability
st.write(f'**Churn Probability:** {prediction_proba:.2f}')

# Display the Churn Prediction Result
if prediction_proba > 0.5:
    st.error('🚨 The customer is likely to churn.')
else:
    st.success('✅ The customer is not likely to churn.')

# Optional: Add a footer or additional explanations if necessary
st.markdown("---")
st.markdown("This app helps predict customer churn based on the provided details.")

