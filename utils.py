import os
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-flash-latest')
else:
    model = None

import time
from google.api_core import exceptions

def generate_with_retry(prompt):
    if not model:
        return "Gemini API key not found. Please set GOOGLE_API_KEY in .env file."
    delays = [10, 20, 30]
    for attempt in range(4):
        try:
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            return "Rate limit reached (15 RPM). Please wait a minute and try again."
        except Exception as e:
            return f"Error generating content: {str(e)}"

def get_prediction_explanation(features_dict, prediction, probability):
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    prompt = f"""
    You are a medical AI assistant in a heart disease risk prediction system. A simple explanation of what this result means
    {features_dict}
    {risk_level} 
    {probability:.2%}
    """
    return generate_with_retry(prompt)

def get_health_chatbot_response(query, patient_context=None):
    context_str = ""
    if patient_context:
        risk_level = "High Risk" if patient_context.get("prediction") == 1 else "Low Risk"
        prob = patient_context.get("probability", 0)
        features = patient_context.get("features", {})
        context_str = f"""
    The patient's current data and prediction results are:
    - Clinical Parameters: {features}
    - Prediction: {risk_level}
    - Heart Disease Probability: {prob:.2%}
    Use this context to give a personalized response.
    """
    prompt = f"""You are a medical AI assistant in a heart disease risk prediction app.
    {context_str}
    The user asks: {query}
    Provide a clear, helpful, and personalized response based on the patient's data if available."""
    return generate_with_retry(prompt)
