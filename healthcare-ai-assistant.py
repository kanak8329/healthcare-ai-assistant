# AI-Powered Virtual Healthcare Assistant
# For Smart Patient Diagnosis & Support

# Install required libraries
!pip install langchain openai pandas scikit-learn joblib flask-cors flask tensorflow kaggle

# Download the dataset from Kaggle
import os
import json
from google.colab import files

# Upload your kaggle.json credentials file
print("Please upload your kaggle.json credentials file to access Kaggle datasets")
uploaded = files.upload()

# Setup Kaggle API credentials
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump(json.loads(list(uploaded.values())[0]), f)
os.chmod('/root/.kaggle/kaggle.json', 600)

# Download the dataset
!kaggle datasets download -d itachi9604/disease-symptom-description-dataset
!unzip disease-symptom-description-dataset.zip

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# Load the dataset
symptom_df = pd.read_csv('Symptom2Disease.csv')
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')
severity_df = pd.read_csv('Symptom-severity.csv')

print(f"Dataset loaded: {len(symptom_df)} records")
symptom_df.head()

# Data preprocessing
# Convert symptom strings to lists of symptoms
symptom_df['Symptoms'] = symptom_df['Symptoms'].apply(lambda x: [s.strip() for s in x.split(',')])

# Get all unique symptoms
all_symptoms = []
for symptoms in symptom_df['Symptoms']:
    all_symptoms.extend(symptoms)
unique_symptoms = sorted(list(set(all_symptoms)))
print(f"Total unique symptoms: {len(unique_symptoms)}")

# Convert symptoms to multi-hot encoding
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_df['Symptoms'])
y = symptom_df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForest model for symptom-based diagnosis
print("Training the diagnostic model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and encoders
joblib.dump(model, 'diagnostic_model.joblib')
joblib.dump(mlb, 'symptom_encoder.joblib')

# Create a symptom-to-index mapping for user input processing
symptom_to_idx = {symptom: i for i, symptom in enumerate(unique_symptoms)}

# Set up LangChain with OpenAI for natural language processing
# You need to set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual API key

# Build an LLM chain for symptom extraction from natural language
symptom_extraction_template = """
You are a medical assistant. Extract all symptoms from the following patient description.
Only return symptoms that are mentioned in the text, in a comma-separated list.
Use the following list of symptoms as a reference: {symptoms_list}

Patient description: {patient_description}

Symptoms:
"""

symptom_extraction_prompt = PromptTemplate(
    input_variables=["symptoms_list", "patient_description"],
    template=symptom_extraction_template
)

# Initialize the LLM for symptom extraction
llm = OpenAI(temperature=0)
symptom_extraction_chain = LLMChain(
    llm=llm,
    prompt=symptom_extraction_prompt
)

# Function to extract symptoms from natural language
def extract_symptoms(patient_description):
    # Get LLM to extract symptoms
    symptoms_comma_list = symptom_extraction_chain.run({
        "symptoms_list": ", ".join(unique_symptoms),
        "patient_description": patient_description
    })
    
    # Process the output
    extracted_symptoms = [s.strip() for s in symptoms_comma_list.split(',')]
    # Filter to only include valid symptoms
    valid_symptoms = [s for s in extracted_symptoms if s in unique_symptoms]
    
    return valid_symptoms

# Function to diagnose based on symptoms
def diagnose(symptoms):
    if not symptoms:
        return "No valid symptoms provided. Please describe your symptoms in more detail."
    
    # Convert symptoms to multi-hot encoding
    symptom_vector = np.zeros(len(unique_symptoms))
    for symptom in symptoms:
        if symptom in symptom_to_idx:
            symptom_vector[symptom_to_idx[symptom]] = 1
    
    # Reshape for prediction
    symptom_vector = symptom_vector.reshape(1, -1)
    
    # Get top 3 disease predictions with probabilities
    proba = model.predict_proba(symptom_vector)[0]
    top_indices = proba.argsort()[-3:][::-1]
    
    results = []
    for idx in top_indices:
        disease = model.classes_[idx]
        probability = proba[idx]
        
        # Get disease description
        description = "No description available."
        if disease in description_df['Disease'].values:
            description = description_df[description_df['Disease'] == disease]['Description'].values[0]
        
        # Get precautions
        precautions = []
        if disease in precaution_df['Disease'].values:
            precaution_row = precaution_df[precaution_df['Disease'] == disease].iloc[0]
            precautions = [precaution_row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(precaution_row[f'Precaution_{i}'])]
        
        results.append({
            'disease': disease,
            'probability': float(probability),
            'description': description,
            'precautions': precautions
        })
    
    return results

# Function to get symptom severity
def get_symptom_severity(symptoms):
    severity_info = []
    for symptom in symptoms:
        if symptom in severity_df['Symptom'].values:
            weight = severity_df[severity_df['Symptom'] == symptom]['weight'].values[0]
            severity = "Low"
            if weight > 5:
                severity = "High"
            elif weight > 3:
                severity = "Medium"
            
            severity_info.append({
                'symptom': symptom,
                'severity': severity,
                'weight': int(weight)
            })
    
    return severity_info

# Create a Flask API
app = Flask(__name__)
CORS(app)

@app.route('/diagnose', methods=['POST'])
def api_diagnose():
    data = request.json
    patient_description = data.get('description', '')
    
    if not patient_description:
        return jsonify({'error': 'Please provide a description of your symptoms'}), 400
    
    # Extract symptoms from the description
    extracted_symptoms = extract_symptoms(patient_description)
    
    if not extracted_symptoms:
        return jsonify({
            'message': 'No recognizable symptoms detected. Please describe your symptoms in more detail.',
            'extracted_symptoms': []
        }), 200
    
    # Get diagnoses
    diagnoses = diagnose(extracted_symptoms)
    
    # Get symptom severity information
    severity_info = get_symptom_severity(extracted_symptoms)
    
    return jsonify({
        'extracted_symptoms': extracted_symptoms,
        'diagnoses': diagnoses,
        'symptom_severity': severity_info
    })

# Create an enhanced response generation function using LangChain
diagnosis_response_template = """
You are an AI healthcare assistant providing information to a patient. 
Based on the following diagnosis information, create a helpful, empathetic response for the patient.
Do NOT claim to be a doctor or provide definitive medical advice. Always suggest consulting a healthcare professional.

Patient's symptoms: {symptoms}
Possible diagnoses: {diagnoses}
Symptom severity information: {severity_info}

Your response should:
1. Acknowledge the symptoms
2. Provide information about the potential conditions
3. Suggest relevant precautions
4. Emphasize the importance of consulting a doctor
5. Be conversational and reassuring

Response:
"""

diagnosis_response_prompt = PromptTemplate(
    input_variables=["symptoms", "diagnoses", "severity_info"],
    template=diagnosis_response_template
)

diagnosis_response_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=diagnosis_response_prompt
)

@app.route('/enhanced-response', methods=['POST'])
def api_enhanced_response():
    data = request.json
    
    symptoms = data.get('symptoms', [])
    diagnoses = data.get('diagnoses', [])
    severity_info = data.get('severity_info', [])
    
    if not symptoms or not diagnoses:
        return jsonify({'error': 'Missing required diagnosis information'}), 400
    
    # Generate enhanced response
    response = diagnosis_response_chain.run({
        "symptoms": ", ".join(symptoms),
        "diagnoses": json.dumps(diagnoses),
        "severity_info": json.dumps(severity_info)
    })
    
    return jsonify({
        'response': response
    })

# Implement a simple demo UI
from IPython.display import HTML, display
import ipywidgets as widgets

def create_ui():
    # Input text box
    text_input = widgets.Textarea(
        value='',
        placeholder='Describe your symptoms here...',
        description='Symptoms:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='150px')
    )
    
    # Output area
    output = widgets.Output()
    
    # Submit button
    button = widgets.Button(
        description='Diagnose',
        disabled=False,
        button_style='primary',
        tooltip='Click to diagnose',
        icon='check'
    )
    
    # Handle button click
    def on_button_clicked(b):
        with output:
            output.clear_output()
            print("Processing your symptoms...")
            
            # Extract symptoms
            symptoms = extract_symptoms(text_input.value)
            print(f"Extracted symptoms: {', '.join(symptoms) if symptoms else 'None detected'}")
            
            if symptoms:
                # Get diagnoses
                diagnoses = diagnose(symptoms)
                
                # Get severity info
                severity = get_symptom_severity(symptoms)
                
                print("\n--- Potential Conditions ---")
                for d in diagnoses:
                    print(f"{d['disease']} (Confidence: {d['probability']:.2f})")
                    print(f"Description: {d['description'][:200]}...")
                    if d['precautions']:
                        print(f"Precautions: {', '.join(d['precautions'])}")
                    print()
                
                print("\n--- Symptom Severity ---")
                for s in severity:
                    print(f"{s['symptom']}: {s['severity']} severity (weight: {s['weight']})")
            else:
                print("Please provide more specific symptom information.")
            
            print("\nDisclaimer: This is not a medical diagnosis. Please consult a healthcare professional.")
    
    button.on_click(on_button_clicked)
    
    # Arrange the UI
    return widgets.VBox([
        widgets.HTML("<h2>AI Healthcare Assistant Demo</h2>"),
        widgets.HTML("<p>Describe your symptoms in detail below:</p>"),
        text_input,
        button,
        widgets.HTML("<h3>Results:</h3>"),
        output
    ])

# Start the Flask app in the background
import threading
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def run_flask_app():
    port = find_free_port()
    print(f"Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port)

# Start Flask in a background thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()

# Display the demo UI
print("Creating interactive demo...")
ui = create_ui()
display(ui)

# Expose the model as a downloadable file for later use
def export_model():
    # Compress the model files
    !zip -r healthcare_assistant_model.zip diagnostic_model.joblib symptom_encoder.joblib

    # Provide download link
    from google.colab import files
    files.download('healthcare_assistant_model.zip')

print("\nTo download the trained model, run: export_model()")

# Add metrics tracking for resume impact
import time
from collections import Counter

# Simulated metrics tracking
query_count = 0
start_time = time.time()
symptom_counts = Counter()
disease_predictions = Counter()

def track_query(symptoms, predicted_diseases):
    global query_count
    query_count += 1
    for symptom in symptoms:
        symptom_counts[symptom] += 1
    for disease in predicted_diseases:
        disease_predictions[disease['disease']] += 1

def get_metrics_summary():
    runtime = time.time() - start_time
    avg_response_time = runtime / max(query_count, 1)
    
    return {
        'total_queries': query_count,
        'avg_response_time': avg_response_time,
        'top_symptoms': dict(symptom_counts.most_common(5)),
        'top_diagnoses': dict(disease_predictions.most_common(5)),
        'runtime': runtime
    }

print("\nTo get performance metrics for your resume, run: get_metrics_summary()")

# Example of how to use the assistant in production code
def production_example():
    print("Example of how to use this assistant in production:")
    print("""
    # Sample code for API integration
    import requests
    
    def get_diagnosis(patient_description):
        response = requests.post(
            'https://your-api-url/diagnose',
            json={'description': patient_description}
        )
        return response.json()
    
    # Example usage
    result = get_diagnosis("I've had a headache, fever, and sore throat for the past two days")
    print(result)
    """)

print("\nTo see integration examples, run: production_example()")

# Print completion message
print("\n✅ AI Healthcare Assistant is ready to use!")
print("Key features implemented:")
print("- Symptom extraction from natural language")
print("- ML-based diagnosis with confidence scores")
print("- Symptom severity assessment")
print("- LLM-enhanced patient responses")
print("- Interactive demo UI")
print("- API endpoints for integration")
print("- Performance metrics tracking")
