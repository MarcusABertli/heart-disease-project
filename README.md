# 🫀 AI-Powered Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?logo=google&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?logo=seaborn&logoColor=white)

An advanced, interactive decision-support system that combines the precision of **Scikit-Learn Machine Learning** with the cognitive capabilities of **Google Gemini AI**. The application empowers users with real-time risk assessment, detailed explainable AI (XAI) reports, and an interactive health companion.

> [!WARNING]
> **Clinical Disclaimer:** This application is designed solely as a decision-support system and **NOT** a medical diagnostic tool. All outputs, suggestions, and chat responses are for educational purposes. Please consult with a certified healthcare professional for actual clinical advice.

---

## 🌟 Key Features

*   **⚡ High-Precision ML Predictions:** Leverages an optimized Machine Learning model trained on standard clinical datasets to output heart disease probability with a risk stratification classification (**Low**, **Medium**, or **High** risk).
*   **🧠 Cognitive Explainable AI (XAI):** Integrated with Google's **Gemini AI (`gemini-flash-lite-latest`)** to parse the patient's individual metrics and generate highly personalized, easy-to-understand risk summaries, highlighting major contributing clinical factors.
*   **📊 Feature Importance Analytics:** Visualizes the key clinical parameters driving the model's predictions using beautiful `matplotlib` and `seaborn` plotting directly on the dashboard.
*   **💬 Interactive Health Companion:** A built-in, context-aware health chatbot positioned on the sidebar to answer questions about heart health, nutrition, or cardiovascular care, customized to the patient's current risk metrics.
*   **📐 Feature Engineering Pipeline:** Implements sophisticated preprocessing pipelines incorporating clinically inspired custom indicators such as Heart Rate Reserve, ST-depression slope products, and age-related maximum heart rate ratios.

---

## 🛠️ Technological Stack

| Layer | Technology | Role / Purpose |
| :--- | :--- | :--- |
| **Frontend & UI** | [Streamlit](https://streamlit.io/) | Interactive web dashboard and real-time user inputs |
| **Generative AI** | [Google Gemini AI API](https://ai.google.dev/) | Contextual clinical explanations and interactive sidebar chatbot |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | Random Forest, Logistic Regression, Gradient Boosting Classifiers |
| **Data Pipelines** | [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) | Preprocessing, feature engineering, and matrix operations |
| **Data Visuals** | [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) | Render feature importance and confusion matrices |
| **Model Storage** | [Joblib](https://joblib.readthedocs.io/) | High-performance serialization of models and scaling pipelines |

---

## 📂 Repository Architecture

```text
heart-disease-project/
├── .devcontainer/         # Dev container environment configuration
├── .env                  # Environment variables template (API keys)
├── .gitignore            # Git exclusion guidelines
├── README.md             # Project documentation and guide
├── app.py                # Main Streamlit web dashboard interface
├── data.csv              # Heart disease clinical dataset for training
├── heart+disease/        # Raw source dataset files (Cleveland, Hungarian, etc.)
├── heart_disease_project.ipynb # Experimental Jupyter research & tuning notebook
├── model.pkl             # Serialized winning model payload
├── requirements.txt      # Python dependencies manifest
└── utils.py              # Google Gemini AI connection & API utilities
```

---

## 🚀 Getting Started & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/MarcusABertli/heart-disease-project.git
cd heart-disease-project
```

### 2. Configure Your Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up API Credentials
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Model Training & Exploration
Model training, parameter optimization, cross-validation, and analysis are detailed and executed interactively inside the Jupyter Notebook:
- Open and run the `heart_disease_project.ipynb` notebook to train and compare Logistic Regression, Random Forest, and Gradient Boosting models, which will output the serialized winning payload to `model.pkl`.

### 6. Run the Application
Launch the Streamlit web server and view your interactive AI dashboard:
```bash
streamlit run app.py
```
By default, the application will spin up on `http://localhost:8501`.

---

## 📐 Clinical Feature Details & Engineering
The following features are collected in the sidebar and utilized for inference:

1.  **Age:** Age of the patient in years.
2.  **Sex:** Gender (Male/Female).
3.  **Chest Pain Type (cp):** Typical angina, atypical angina, non-anginal pain, or asymptomatic.
4.  **Resting Blood Pressure (trestbps):** Resting blood pressure in mm Hg.
5.  **Serum Cholesterol (chol):** Cholesterol level in mg/dl.
6.  **Fasting Blood Sugar (fbs):** Fasting blood sugar > 120 mg/dl (True/False).
7.  **Resting ECG (restecg):** Resting electrocardiographic results.
8.  **Max Heart Rate (thalach):** Maximum heart rate achieved during exercise.
9.  **Exercise Induced Angina (exang):** Pain induced by exercise (Yes/No).
10. **ST Depression (oldpeak):** ST depression induced by exercise relative to rest.
11. **Slope:** The slope of the peak exercise ST segment.
12. **Major Vessels (ca):** Number of major vessels (0-3) colored by fluoroscopy.
13. **Thal (thal):** Thalassemia status (Normal, Fixed defect, Reversible defect).

### 🛠️ Engineered Features

| Feature | Formula | What it means |
|---|---|---|
| `heart_rate_reserve` | `thalach / (220 - age)` | Max heart rate as a fraction of age-predicted max |
| `age_thalach_ratio` | `age / (thalach + 1)` | Age relative to max heart rate |
| `oldpeak_slope` | `oldpeak * slope` | ST depression combined with ST slope |
| `cp_exang` | `cp * exang` | Chest pain type combined with exercise angina |
| `age_binned` | `0 = Under 40, 1 = 40–55, 2 = Over 55` | Age group index |
| `trestbps_high` | `1 if trestbps > 140, else 0` | High resting blood pressure flag |
