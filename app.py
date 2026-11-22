import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Disease Prediction App", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.markdown("""
This application predicts the likelihood of heart disease based on medical attributes.
""")

@st.cache_resource
def load_models():
    model_path = os.path.join("models", "final_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure you have trained the model first.")
        return None, None

model, scaler = load_models()

st.header("Patient Medical Data")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4],
                      format_func=lambda x: f"{x}: " + ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x-1])
    trestbps = st.number_input("Resting Blood Pressure (trestbps) [mm Hg]", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (chol) [mg/dl]", min_value=100, max_value=600, value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2],
                           format_func=lambda x: f"{x}: " + ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

with col3:
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[1, 2, 3],
                         format_func=lambda x: f"{x}: " + ["Upsloping", "Flat", "Downsloping"][x-1])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0.0, 1.0, 2.0, 3.0])
    thal = st.selectbox("Thalassemia (thal)", options=[3.0, 6.0, 7.0],
                        format_func=lambda x: {3.0: "3: Normal", 6.0: "6: Fixed Defect", 7.0: "7: Reversable Defect"}[x])

if st.button("PREDICT", type="primary"):
    if model is None or scaler is None:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("Prediction Results")

        if prediction == 1:
            st.error("Model prediction: Positive")
        else:
            st.success("Model prediction: Negative")

        st.info(f"Confidence (Probability of Disease): {probability:.2%}")

        st.warning("⚠️ **Disclaimer:** This result is for educational/research purposes and should not be considered a medical diagnosis. Please consult a healthcare professional for clinical advice.")
