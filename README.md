🏥 AI-Powered Virtual Healthcare Assistant
A smart AI-based assistant for patient symptom analysis, disease prediction, severity assessment, and empathetic healthcare support.
It combines Machine Learning (Random Forest), LLMs (OpenAI API via LangChain), and Flask APIs into an easy-to-use solution.

✨ Features
✅ Symptom extraction from natural language descriptions

✅ Disease prediction with confidence scores

✅ Symptom severity analysis based on weight

✅ Empathetic patient responses using LLM (OpenAI)

✅ Interactive demo UI inside Jupyter/Colab

✅ REST API endpoints for easy integration

✅ Downloadable trained model for deployment

✅ Performance tracking for metrics

🛠️ Setup Instructions
1. Install Dependencies
bash
Copy
Edit
pip install langchain openai pandas scikit-learn joblib flask-cors flask tensorflow kaggle
2. Dataset Download (from Kaggle)
Upload your kaggle.json API key.

The script downloads and unzips the dataset automatically.

3. Set OpenAI API Key
Inside the script, replace:

python
Copy
Edit
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
with your actual OpenAI key.

🚀 How It Works
Train a RandomForestClassifier on symptoms vs diseases.

Extract Symptoms from free-text patient descriptions using an LLM.

Diagnose Diseases based on detected symptoms.

Assess Symptom Severity (Low/Medium/High).

Generate Empathetic Responses for patients using a conversational template.

Expose APIs for diagnosis and enhanced response generation.

Interactive UI to test everything live.

🧪 API Endpoints
POST /diagnose
Input:

json
Copy
Edit
{ "description": "I have a sore throat and fever" }
Output: extracted symptoms, predicted diseases, symptom severity.

POST /enhanced-response
Input:

json
Copy
Edit
{
  "symptoms": ["sore throat", "fever"],
  "diagnoses": [...],
  "severity_info": [...]
}
Output: a conversational healthcare assistant response.

📈 Metrics Tracking
Track:

Total queries

Average response time

Top queried symptoms

Top predicted diseases

Call:

python
Copy
Edit
get_metrics_summary()
🎯 Running the Project
Works best in Google Colab or Jupyter Notebooks.

Runs a background Flask server + an interactive demo UI.

Example production integration code is also provided.

📦 Model Export
Save and download trained models for future use:

python
Copy
Edit
export_model()
This creates a downloadable healthcare_assistant_model.zip.

📋 Example Usage (Production)
python
Copy
Edit
import requests

def get_diagnosis(patient_description):
    response = requests.post(
        'https://your-api-url/diagnose',
        json={'description': patient_description}
    )
    return response.json()

result = get_diagnosis("I've had a headache, fever, and sore throat for the past two days")
print(result)
⚠️ Disclaimer
This tool is for educational purposes only.
It does not replace professional medical advice.
Always consult qualified healthcare professionals for any medical issues.

🧠 Technologies Used
🧪 Scikit-learn

🔥 Flask

🧠 OpenAI (via LangChain)

📚 Pandas, NumPy

🏥 TensorFlow (future extensions possible)

🛠️ Joblib

🐍 Python
