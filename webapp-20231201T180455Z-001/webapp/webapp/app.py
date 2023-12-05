

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import numpy as np
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from neo4j import GraphDatabase
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

minor_diseases=['Abscess','Acid reflux','Alcohol intoxication','Bed bug bite','Common cold',
'Bronchitis','Cataracts','Atopic dermatitis','Carpal tunnel syndrome','Chicken pox','Concussion',
'Anemia','Diverticulosis','Fibroids','Gallstones','Gastritis','Gout','Groin Hernia','Hiatal Hernia',
'High Cholesterol','Hives','Hypoglycemia','Hypothyroidism','Insect Bites and Stings','Iron Deficiency Anemia',
'Irritable Bowel Syndrome','Kidney Stone','Low Back Pain','Middle Ear Infection','Migraine','Pink Eye',
'Seasonal Allergies','Sebaceous Cyst','Sinusitis','Sprained Ankle','Sprained Knee','Sprained Wrist',
'Stomach Flu','Tension Headache','Vitamin D Deficiency','Deglutition disorder','Hyperlipidemia',
'Incontinence','Neuropathy','Paranoia','Ulcer peptic','Benign prostatic hypertrophy','Gastroenteritis',
'Hemiparesis','Thrombocytopaenia','Dehydration','Hemorrhoids','spasm bronchial','hernia hiatal',
'migraine disorders','tachycardia sinus','allergy','GERD','hypertension','cervical spondylosis',
'chicken pox','common cold','dimorphic hemorrhoids (piles)','varicose veins','hypothyroidism',
'osteoarthritis','arthritis','(vertigo) paroxysmal positional vertigo','acne','urinary tract infection',
'psoriasis','impetigo']


moderate_diseases=['Alcoholism','Anaphylaxis','Aphasia','Appendicitis','Arthritis','Asthma',
'Atrial fibrillation', 'Bipolar disorder','Bone fracture','Bone infection','Burn',
'Cellulitis','Cervical radiculopathy','Chlamydia','Cholecystitis','Chronic pain','Concussion','Dementia',
'Dementia','Diverticulitis','Drug Withdrawal','Ectopic Pregnancy','Endometriosis','Epididymitis',
'Epilepsy','Fibromyalgia','Gastroparesis','Glaucoma','Goiter','Hepatitis A','Hepatitis B','Hepatitis C',
'High Blood Pressure','Hyperthyroidism','Hyponatremia','Kidney Infection','Lyme Disease',
'Major Depression','Meningitis','Miscarriage','Mono','Obesity','Osteoarthritis',
'Osteoporosis','Plantar Fasciitis','Polycystic Ovary Syndrome','Psoriasis',
'Sciatica','Sleep Apnea','Spondylosis','Stomach Ulcer','Thyroid Nodules','Type 2 Diabetes',
'Upper Respiratory Infection','Urinary Tract Infection','Depression Mental','Hypercholesterolemia',
'Anxiety State','Gastroesophageal reflux disease','Deep vein thrombosis','Transient ischemic attack',
'Hepatitis','Colitis','Oral candidiasis','Osteomyelitis','Hernia','Fibroid tumor','Parkinson disease',
'Hypertension pulmonary','Personality disorder','obesity morbid','pyelonephritis','chronic alcoholic intoxication',
'delirium','influenza','dependence','cholelithiasis','biliary calculus','ileus','affect labile',
'fungal infection','drug reaction','peptic ulcer disease','diabetes','gastroenteritis','bronchial asthma',
'jaundice','dengue','hypoglycemia','psoriasis']

major_diseases=['Acute renal failure','Aneurysm','Angina','Arrhythmia','Borderline personality disorder',
'Celiac disease','Chronic kidney disease','Chronic obstructive pulmonary disease','Cirrhosis of the liver',
'Deep vein thrombosis','Cardiomyopathy','Encephalitis','Encephalopathy','Esophageal Cancer',
'Heart Attack','HIV/AIDS','Liver Cancer','Lung Cancer','Lupus','Melanoma','Multiple Sclerosis','Myasthenia Gravis',
'Pneumonia','Prostate Cancer','Pulmonary Embolism','Pulmonary Hypertension','Rheumatoid Arthritis',
'Schizophrenia','Shingles','Type 1 Diabetes','Ulcerative Colitis','Hypertensive Disease','Diabetes',
'Coronary Arteriosclerosis','Coronary Heart Disease','Failure Heart Congestive','Chronic Obstructive Airway Disease',
'Insufficiency Renal','Degenerative Polyarthritis','Malignant Neoplasms','Primary Malignant Neoplasm',
'Septicemia','Systemic infection','Neoplasm','Embolism pulmonary','Chronic kidney failure',
'Carcinoma','Hepatitis C','Peripheral vascular disease','Psychotic disorder','Bipolar disorder',
'Ischemia','Cirrhosis','Exanthema','Kidney failure acute','Mitral valve insufficiency','Adenocarcinoma',
'Paroxysmal dyspnea','Malignant neoplasm of prostate','Carcinoma prostate','Edema pulmonary',
'Lymphatic diseases','Stenosis aortic valve','Malignant neoplasm of breast','Carcinoma breast',
'Overload fluid','Failure kidney','Sickle cell anemia','Failure heart','Pneumocystis carinii pneumonia',
'Hepatitis B','Lymphoma','Tricuspid valve insufficiency','Candidiasis','Kidney disease',
'Neoplasm metastasis','Malignant tumor of colon','Carcinoma colon','Respiratory failure','Malignant neoplasm of lung',
'Carcinoma of lung','Manic disorder','Suicide attempt','Primary carcinoma of the liver cells',
'emphysema pulmonary','endocarditis','effusion pericardial','pericardial effusion body substance',
'hyperbilirubinemia','thrombus','pancytopenia','adhesion','decubitus ulcer','chronic cholestasis',
'AIDS','paralysis','malaria','typhoid','alcoholic hepatitis','tuberculosis','pneumonia',
'heart attack']


extreme_diseases=['Delusion','Bladder cancer','Brain tumor','Breast cancer','Cardiac arrest',
'Cervical cancer','Cholangiocarcinoma','Colon cancer','Congestive heart failure',
'Coronary artery disease','Endometrial Cancer','Multiple Myeloma','Neutropenia',
'Ovarian Cancer','Ovarian Cyst','Pancreatitis','Pleural Effusion','Pneumothorax',
'Post Traumatic Stress Disorder','Psychosis','Retinal Detachment','Rhabdomyolysis','Sepsis',
'Sickle Cell Anemia','Strep Throat','Stroke','Thoracic Radiculopathy','Thyroid Cancer',
'Traumatic Brain Injury','Accident Cerebrovascular','Myocardial Infarction','Infection',
'Infection Urinary Tract','Confusion','Acquired Immuno-Deficiency Syndrome','HIV',
'HIV Infections','Ketoacidosis diabetic','Tonic-clonic epilepsy','Tonic-clonic seizures']

class_mapping = {}

# Populate the dictionary with disease-class mappings
for disease in minor_diseases:
    class_mapping[disease] = 'minor_diseases'
for disease in moderate_diseases:
    class_mapping[disease] = 'moderate_diseases'
for disease in major_diseases:
    class_mapping[disease] = 'major_diseases'
for disease in extreme_diseases:
    class_mapping[disease] = 'extreme_diseases'

class_probs = {
    'minor_diseases': 0,
    'moderate_diseases': 0,
    'major_diseases': 0,
    'extreme_diseases': 0
}

app = Flask(__name__)

symptoms_list=[]
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

print(symptoms_list)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data['text']

    df = pd.DataFrame(pipe(text))
    df = df.drop(['score', 'start', 'end'], axis=1)
    symptoms_df = df[df["entity_group"] == "Sign_symptom"]
    if not symptoms_df.empty:
        unique_symptoms = symptoms_df["word"].unique()

        for symptom in unique_symptoms:
            symptoms_list.append(symptom)


    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].astype(float)

    return df.to_json(orient='records')

@app.route('/finalize', methods=['POST'])
def finalize():
    global symptoms_list
    data = request.get_json()
    frontend_symptoms = data['symptoms']

    # Append frontend symptoms to the global symptoms list
    symptoms_list.extend(frontend_symptoms)
    unique_symptoms_set = set(symptoms_list)
    symptoms_list = list(unique_symptoms_set)
    print("Final Symptoms List:", symptoms_list)
    
    # return "Symptoms processed successfully"
    return jsonify(symptoms_list)


@app.route('/process_symptoms', methods=['POST'])
def process_symptoms():
    data = request.get_json()
    symptoms = data['symptoms']
    for i in symptoms:
        symptoms_list.append(i)
  
    return jsonify({'receivedSymptoms': symptoms})


def diseaseprobability(df_result):
    for index, row in df_result.iterrows():
    
        if row['Disease'] in class_mapping:
        # If the disease is found in the mapping, get its class
            disease_class = class_mapping[row['Disease']]
            class_probs[disease_class] += row['Probability']
        # print(disease_class)
        else:
        # If the disease is not found in the mapping, skip this iteration
            continue
    total_prob = sum(class_probs.values())
    class_probs_normalized = {k: v / total_prob for k, v in class_probs.items()}

# Output the class probabilities
    print(class_probs_normalized)

from flask import render_template_string




def predict_top_diseases(user_symptoms, user_severities, mlb, nb_model, ann_model, label_encoder):
   
    # One-hot encode the input symptoms
    symptoms_encoded = mlb.transform([user_symptoms])
    symptom_presence = pd.DataFrame(symptoms_encoded, columns=mlb.classes_)

    # Create a DataFrame for symptom severity
    symptom_severity = pd.DataFrame(0, index=np.arange(1), columns=mlb.classes_)
    for symp, weight in zip(user_symptoms, user_severities):
        if symp in mlb.classes_:
            symptom_severity.at[0, symp] = weight
    
    # Combine symptom presence and severity
    combined_input = pd.concat([symptom_presence, symptom_severity.add_suffix('_severity')], axis=1)

    # Get Naive Bayes probabilities
    nb_probabilities = nb_model.predict_proba(combined_input)

    # Prepare ANN input
    ann_input = np.hstack((combined_input, nb_probabilities))

    # Get ANN predictions
    ann_predictions = ann_model.predict(ann_input)[0]

    # Get indices of top 5 predictions
    top_indices = np.argsort(ann_predictions)[-5:][::-1]
    top_probabilities = ann_predictions[top_indices]
    top_diseases = label_encoder.inverse_transform(top_indices)

    return list(zip(top_diseases, top_probabilities))


    
formatted_symptoms=[]

@app.route('/process_symptoms_severity', methods=['POST'])
def process_symptoms_severity():
    data = request.get_json()
    symptoms_severity = data['symptomsSeverity']

   
    print("what we need",symptoms_severity)
    
    symptoms = []
    severities = []

# Extract symptoms and severities
    for item in symptoms_severity:
        symptoms.append(item['symptom'])
        severities.append(int(item['severity'])) 
    
    loaded_gnb = joblib.load('naive_bayes_model.pkl')

    loaded_ann_model = load_model('ann_model.h5')

    loaded_mlb = joblib.load('multilabelbinarizer.pkl')

    loaded_label_encoder = joblib.load('label_encoder.pkl')

    # Example usage with loaded models
    top_diseases_predictions = predict_top_diseases(symptoms, severities, loaded_mlb, loaded_gnb, loaded_ann_model, loaded_label_encoder)

    print(top_diseases_predictions)
    disease_names = [disease for disease, _ in top_diseases_predictions]

    # Send the disease names back to the frontend
    return jsonify({'diseases': disease_names})








    

    

    # return jsonify({'status': 'Processed symptoms with severity'})

if __name__ == '__main__':
    app.run(debug=True)