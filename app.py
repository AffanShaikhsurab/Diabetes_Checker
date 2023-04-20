import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras as kr
import pickle
from sklearn.preprocessing import MinMaxScaler

st.title("Diabetes Checker")
age = st.slider("Age")
gender = st.radio("Gender", options=["Male", "Female"])
hypertension = st.checkbox("Hypertension")
heart_disease = st.checkbox("Heart Disease")
smoking_history = st.radio("Smoking History", options=["Never", "Current"])
bmi = st.number_input("BMI")
HbA1c_level = st.number_input("HbA1c Level")
blood_glucose_level = st.number_input("Blood Glucose Level")
ok = st.button("Start Checking")

with open('transform_value.pk1', 'rb') as f:
    scaler = pickle.load(f)

model = kr.models.load_model('model.h5')
gender_male = 0
gender_female = 0
gender_other = 0
smoking_history_no_info = 0
smoking_history_current = 0
smoking_history_ever = 0
smoking_history_former = 0
smoking_history_never = 0
smoking_history_not_current = 0

if ok:
    if gender == "Male":
        gender_male = 1
    elif gender == "Female":
        gender_female = 1

    if smoking_history == "never":
        smoking_history_never = 1
        smoking_history_not_current = 1
    elif smoking_history == "current":
        smoking_history_current = 1
        smoking_history_ever = 1
        smoking_history_former = 1

    input_data = np.array([age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level ,gender_female , gender_male , gender_other , smoking_history_no_info,
                           smoking_history_current , smoking_history_ever , smoking_history_former , smoking_history_never , smoking_history_not_current]).reshape(1,-1)
# 
    transformed_input = scaler.transform(input_data)
    prediction = model.predict(transformed_input)


    if prediction > 0.5:
        output = "You might have a chance of diabetes."
    else:
        output = "You might not have a chance of diabetes."

    st.header(output)
