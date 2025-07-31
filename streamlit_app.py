#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import numpy as np
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import datetime

## CONFIG 
st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")

## LOAD MODEL
model = pickle.load(open("heart_attack_xgb_model.sav", 'rb'))

## VISITOR COUNTER 
counter_file = "heart_visitors.pkl"

if os.path.exists(counter_file):
    with open(counter_file, 'rb') as f:
        visitor_data = pickle.load(f)
else:
    visitor_data = {'count': 0, 'timestamps': []}

if 'counted' not in st.session_state:
    st.session_state['counted'] = True
    visitor_data['count'] += 1
    visitor_data['timestamps'].append(str(datetime.datetime.now()))
    with open(counter_file, 'wb') as f:
        pickle.dump(visitor_data, f)

## SIDEBAR MENU 
with st.sidebar:
    selected = option_menu("Main Menu", 
                           ["Welcome", "Prediction", "FAQ", "Disclaimer", "Analytics"], 
                           icons=['house', 'activity', 'question-circle', 'exclamation-circle', 'bar-chart'],
                           menu_icon="cast", default_index=0)

## WELCOME PAGE
if selected == "Welcome":
    st.markdown("<h1 style='text-align: center; color: #28a745;'> Heart Attack Risk Estimator</h1>", unsafe_allow_html=True)
    st.success(f"👥 *Total Visitors:* {visitor_data['count']}")
    st.write("""
This application leverages a trained machine learning model (XGBoost) to assess an individual's probability of experiencing a heart attack using key health indicators such as age, blood pressure, heart rate, and cardiac biomarkers.

It is designed to:
- Promote early risk awareness
- Empower users with data-driven insights
- Encourage timely consultation with healthcare professionals

👉 Use the sidebar to start!
    """)

## PREDICTION PAGE
elif selected == "Prediction":
    st.markdown(
        "<h2 style='text-align: center; color: #28a745;'>🫪 Predict Heart Attack Risk</h2>", 
        unsafe_allow_html=True
    )
    st.write("Please fill in the following medical details:")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=0, step=1)
            gender = st.selectbox("Gender", options=["Female", "Male"])
            gender_code = 1 if gender == "Male" else 0

        with col2:
            heart_rate = st.number_input("Heart Rate", min_value=0, step=1)
            systolic_bp = st.number_input("Systolic Blood Pressure", min_value=0, step=1)
            diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=0, step=1)

        with col3:
            blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=0.0, step=0.1)
            ck_mb = st.number_input("CK-MB Level", min_value=0.0, step=0.1)
            troponin = st.number_input("Troponin Level", min_value=0.0, step=0.01)

        submit = st.form_submit_button("🔍 Predict Risk")

    if submit:
        input_features = np.array([[age, gender_code, heart_rate, systolic_bp, diastolic_bp,
                                    blood_sugar, ck_mb, troponin]])
        probability = model.predict_proba(input_features)[0][1]
        st.success(f"🧠 Estimated Risk of Heart Attack: **{probability:.2%}**")

        if probability >= 0.7:
            st.error("⚠ High risk! Immediate medical consultation is recommended.")
        elif probability >= 0.4:
            st.warning("⚠ Moderate risk. Consider lifestyle changes and medical advice.")
        else:
            st.success("✅ Low risk. Keep up the good habits!")

        ## RECOMMENDATIONS
        st.markdown("### 💡 Personalized Health Recommendations")

        if age > 50:
            st.info("👉 Age above 50 is a known risk factor. Ensure regular health screenings.")

        if systolic_bp > 140 or diastolic_bp > 90:
            st.warning("⚠ High blood pressure detected. Limit salt, manage stress, and monitor regularly.")

        if heart_rate < 60:
            st.info("💤 Low resting heart rate. If not athletic, consult your doctor.")
        elif heart_rate > 100:
            st.warning("⚠ Elevated heart rate. Could indicate stress, fever, or arrhythmia.")

        if blood_sugar >= 7.0:
            st.warning("⚠ High blood sugar can lead to cardiovascular issues. Manage your diet and activity.")

        if ck_mb > 5:
            st.warning("⚠ Elevated CK-MB suggests possible cardiac stress. Medical review is recommended.")

        if troponin > 0.04:
            st.error("⚠ Troponin level is high. This could indicate cardiac injury. See a cardiologist.")

        st.success("✅ Stay active, eat heart-healthy foods, avoid smoking, and get regular checkups.")

# FAQ PAGE
elif selected == "FAQ":
    st.markdown(
        "<h2 style='text-align: center; color: #28a745;'>❓ Frequently Asked Questions</h2>",
        unsafe_allow_html=True
    )

    with st.expander("📌 What does this app do?"):
        st.write("This app predicts the probability of heart attack using machine learning based on key health parameters.")

    with st.expander("📌 Can this tool replace a doctor?"):
        st.write("No. It's designed for risk awareness and early detection only. Always consult a qualified medical professional.")

    with st.expander("📌 Is my personal data stored?"):
        st.write("No. Your input data is processed only on your device and is never stored or shared.")

    with st.expander("📌 What is CK-MB and Troponin?"):
        st.write("CK-MB and Troponin are enzymes and proteins released into the blood when the heart muscle is damaged.")

    with st.expander("📌 What model is used in this app?"):
        st.write("This app uses the **XGBoost (Extreme Gradient Boosting)** model, a powerful and efficient machine learning algorithm. "
                 "It was selected after outperforming several other models based on key performance metrics such as **accuracy**, **recall**, "
                 "**precision**, **AUC**, and **F1 score** during model evaluation.")

    with st.expander("📌 Is this app suitable for regular screening?"):
        st.write("Yes, but it must not replace regular clinical checkups.")

    with st.expander("📌 Who can benefit from using this app?"):
        st.write("Anyone at potential risk, especially those over 40, sedentary, or with family history of heart disease.")

# DISCLAIMER PAGE
elif selected == "Disclaimer":
    st.markdown(
        "<h2 style='text-align: center; color: #28a745;'>⚠ Disclaimer</h2>",
        unsafe_allow_html=True
    )
    st.warning("""
This tool is for informational purposes only and is not a substitute for professional healthcare.
Predictions should not be used to make medical decisions without a physician's evaluation.
""")

# ANALYTICS PAGE
elif selected == "Analytics":
    st.markdown(
        "<h2 style='text-align: center; color: #28a745;'>📊 Visitor Analytics</h2>",
        unsafe_allow_html=True
    )

    st.info(f"👥 *Total Visitors:* {visitor_data['count']}")

    if visitor_data['timestamps']:
        st.write("### 🕒 Visitor Log")
        st.dataframe(visitor_data['timestamps'])
