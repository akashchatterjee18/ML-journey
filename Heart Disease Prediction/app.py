import streamlit as st
import pandas as pd
import joblib

model = joblib.load('Logistic Regression_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')

st.title("Heart Disease Prediction")
st.markdown("Provide the following Details to predict the risk of heart disease.")

age = st.slider("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA","NAP","TA","ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80,200,120)
cholesterol = st.number_input("Cholesterol (mm/dL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0,1])
resting_ecg = st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr = st.slider("Max Heart Rate",60,220,120)
exercise_angina = st.selectbox("Excercise-Induced Angina", ["Y","N"])
oldpeak = st.slider("Oldpeak(ST Depression)",0.0,6.0,3.0)
st_slope = st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_scaled = scaler.transform(input_df)

    probabilities = model.predict_proba(input_scaled)[0]

    no_disease_probability = probabilities[0]
    disease_probability = probabilities[1]

    disease_percent = disease_probability * 100
    no_disease_percent = no_disease_probability * 100

    if disease_probability >= 0.5:
        st.error(f"Heart Disease Probability: {disease_percent:.2f}%")
        st.success(f"No Heart Disease Probability: {no_disease_percent:.2f}%")
    else:
        st.success(f"No Heart Disease Probability: {no_disease_percent:.2f}%")
        st.error(f"Heart Disease Probability: {disease_percent:.2f}%")