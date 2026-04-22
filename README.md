# AI-Powered Heart Disease Risk Predictor

## Overview

The **AI-Powered Heart Disease Risk Predictor** is a machine learning-based decision-support system built with Python, Streamlit, and Scikit-Learn. It takes patient clinical parameters as input to predict the probability of heart disease using a trained Random Forest model. 

In addition to traditional ML predictions, this application leverages Google's **Gemini AI** to generate clear, human-readable explanations of the results, highlighting key contributing factors, and offering an integrated health chatbot for general inquiries.

> ⚠️ **Disclaimer:** This is a decision-support system and NOT a medical diagnostic tool. Please consult with a healthcare professional for clinical advice.

## Features

- **Clinical Data Input:** Sidebar forms for users to input 13 critical clinical parameters (Age, Sex, Chest Pain Type, Resting BP, Cholesterol, etc.).
- **Machine Learning Prediction:** Uses a Random Forest Classifier trained on clinical data to provide a rapid risk assessment (High Risk vs. Low Risk) and exact probability percentage.
- **AI-Generated Explanations:** Utilizes Google's `gemini-2.0-flash` model to analyze the user's data and the model's prediction to provide personalized context, highlighting which health factors contributed most to the assessment.
- **Feature Importance Visualization:** Displays a bar chart ranking the most critical clinical features influencing the prediction.
- **Interactive Health Chatbot:** A built-in sidebar chatbot capable of answering general heart-health queries (e.g., "How to lower cholesterol?").

## Tech Stack

- **Frontend:** Streamlit
- **Machine Learning:** Scikit-learn (Random Forest Classifier, Logistic Regression), Pandas, NumPy
- **Generative AI:** Google GenAI (`gemini-2.0-flash`)
- **Visualizations:** Matplotlib, Seaborn

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MarcusABertli/heart-disease-project.git
   cd heart-disease-project
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your Google Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

4. **Train the Model:**
   Before running the app, train the machine learning model to generate the `model.pkl` file:
   ```bash
   python train_model.py
   ```

5. **Run the Application:**
   Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

## Repository Structure

- `app.py`: Main Streamlit application containing the UI and logic for prediction.
- `train_model.py`: Script to preprocess the data, train the Random Forest and Logistic Regression models, and save the best model (`model.pkl`).
- `utils.py`: Helper functions for integrating the Google Gemini AI for explanations and the chatbot functionality.
- `data.csv`: The heart disease dataset used for training the model.
- `requirements.txt`: Python dependencies.
