import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None

import time
from google.api_core import exceptions

def generate_with_retry(prompt):
    """
    Helper function to generate content with exponential backoff to handle 15 RPM free tier limits.
    """
    if not model:
        return "Gemini API key not found. Please set GOOGLE_API_KEY in .env file."
        
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            return response.text
        except exceptions.ResourceExhausted:
            if attempt < 2:
                time.sleep(2 ** (attempt + 2))  # Wait 4, 8 seconds
                continue
            return "Rate limit reached (15 RPM). Please wait a moment and try again."
        except Exception as e:
            return f"Error generating content: {str(e)}"

def get_prediction_explanation(features_dict, prediction, probability):
    """
    Generate a human-like explanation for the heart disease risk prediction.
    """
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    
    prompt = f"""
    You are a medical AI assistant in a heart disease risk prediction system. 
    The system predicts if a patient is at high or low risk based on clinical data.
    
    IMPORTANT: 
    1. Do NOT claim to be a doctor. 
    2. State clearly this is a decision-support system, not a diagnosis.
    3. Be encouraging and provide general health advice.
    
    Patient Clinical Features:
    {features_dict}
    
    Prediction: {risk_level} (Probability of Heart Disease: {probability:.2%})
    
    Please provide:
    - A simple explanation of what this result means.
    - Potential contributing factors from the patient's data (e.g., high blood pressure, cholesterol, age).
    - General lifestyle recommendations for heart health.
    - A clear medical disclaimer.
    """
    return generate_with_retry(prompt)

def get_health_chatbot_response(query):
    """
    General chatbot for health-related questions.
    """
    prompt = f"""
    The user is asking a health-related question in the context of a heart disease risk prediction app.
    
    Question: {query}
    
    Guidelines:
    - Provide a concise, professional, and helpful response.
    - Always include a disclaimer that this is not medical advice.
    - If the question is not about health, politely redirect them.
    - Keep it focused on general heart health if appropriate.
    """
    return generate_with_retry(prompt)
