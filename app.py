import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_prediction_explanation, get_health_chatbot_response
st.set_page_config(page_title="Heart Disease Risk Prediction System", layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #f5f7f9;
}
.main-title {
    color: #2c3e50;
    text-align: center;
    font-size: 3rem;
    padding: 1rem;
}
.prediction-box {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    color: white;
}
.low-risk { background-color: #27ae60; }
.high-risk { background-color: #c0392b; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">AI-Powered Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.warning("⚠️ **Disclaimer**: This is a decision-support system, NOT a medical diagnostic tool. Please consult with a healthcare professional for clinical advice.")
st.sidebar.header("📋 Patient Clinical Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age (years)", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], 
                             format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholestoral (mg/dl)", min_value=50, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox("Slope of the peak exercise ST segment (slope)", options=[1, 2, 3])
    ca = st.sidebar.selectbox("Number of major vessels colored (ca)", options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thal (thal)", options=[3, 6, 7], 
                                format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}[x])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("📊 Patient Inputs Summary")
st.write(input_df)

if st.button("🚀 Predict Heart Disease Risk"):
    if not os.path.exists('model.pkl'):
        st.error("❌ Model file 'model.pkl' not found. Please run 'python train_model.py' first.")
    else:
        model_data = joblib.load('model.pkl')
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        input_data = input_df[features]
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        st.subheader("🔮 Prediction Result")
        risk_class = "high-risk" if prediction == 1 else "low-risk"
        risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"
        
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2 style="margin:0;">Risk Level: {risk_label}</h2>
            <p style="margin:0; font-size:1.2rem;">Heart Disease Probability: {probability:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("💡 AI Explanation & Recommendations")
        with st.spinner("Generating AI explanation..."):
            explanation = get_prediction_explanation(input_df.to_dict('records')[0], prediction, probability)
            st.write(explanation)
        st.subheader("📈 Top Contributing Factors")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(features)[indices], ax=ax, palette="viridis")
        ax.set_title("Feature Importance Influencing Prediction")
        st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask a Heart Health Question")
user_query = st.sidebar.text_input("Example: How to lower cholesterol?")
if user_query:
    with st.spinner("AI Chatbot thinking..."):
        response = get_health_chatbot_response(user_query)
        st.sidebar.write(response)
