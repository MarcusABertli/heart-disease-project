import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_prediction_explanation, get_health_chatbot_response

@st.cache_data(show_spinner=False)
def fetch_explanation(input_dict, pred, prob):
    return get_prediction_explanation(input_dict, pred, prob)

@st.cache_data(show_spinner=False)
def fetch_chatbot_response(query, patient_context=None):
    return get_health_chatbot_response(query, patient_context)

st.set_page_config(page_title="Heart Disease Risk Prediction System", layout="wide")
st.markdown("""
<style>
.main-title {
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
        with st.spinner("Analyzing data and generating AI explanation..."):
            model_data = joblib.load('model.pkl')
            model = model_data['model']
            features = model_data['features']

            input_fe = input_df.copy()
            input_fe['heart_rate_reserve'] = input_fe['thalach'] / (220 - input_fe['age'])
            input_fe['age_thalach_ratio'] = input_fe['age'] / (input_fe['thalach'] + 1)
            input_fe['oldpeak_slope'] = input_fe['oldpeak'] * input_fe['slope']
            input_fe['cp_exang'] = input_fe['cp'] * input_fe['exang']
            input_fe['age_binned'] = pd.cut(input_fe['age'], bins=[0, 40, 55, 100], labels=[0, 1, 2]).astype(int)
            input_fe['trestbps_high'] = (input_fe['trestbps'] > 140).astype(int)

            input_data = input_fe[features]
            
            if 'scaler' in model_data and model_data['scaler'] is not None:
                scaler = model_data['scaler']
                input_processed = scaler.transform(input_data)
            else:
                input_processed = input_data
                
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0][1]
            
            st.session_state['prediction'] = prediction
            st.session_state['probability'] = probability
            
            explanation = fetch_explanation(input_df.to_dict('records')[0], prediction, probability)
            st.session_state['explanation'] = explanation
            
            original_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
            feat_imp = feat_imp[feat_imp['Feature'].isin(original_features)]
            feat_imp = feat_imp.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax, palette="viridis")
            ax.set_title("Feature Importance Influencing Prediction")
            st.session_state['fig'] = fig

if 'prediction' in st.session_state:
    prediction = st.session_state['prediction']
    probability = st.session_state['probability']
    explanation = st.session_state['explanation']
    fig = st.session_state['fig']
    
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
    st.write(explanation)
    
    st.subheader("📈 Top Contributing Factors")
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Ask a Heart Health Question")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_container = st.sidebar.container()

for msg in st.session_state.chat_history:
    with chat_container.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.sidebar.chat_input("e.g. Lower cholesterol?")

if user_query:
    with chat_container.chat_message("user"):
        st.write(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    patient_context = None
    if 'prediction' in st.session_state:
        patient_context = {
            "prediction": st.session_state['prediction'],
            "probability": st.session_state['probability'],
            "features": input_df.to_dict('records')[0]
        }
    
    with chat_container.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = fetch_chatbot_response(user_query, patient_context)
            st.write(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
