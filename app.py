import json
import torch
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd

from nnet import NeuralNet
from nltk_utils import bag_of_words
from flask import Flask, render_template, request, jsonify

# ------------------------------
# Random seed
# ------------------------------

random.seed(datetime.now().timestamp())

device = torch.device('cpu')

# ------------------------------
# Load NLP Intent Model
# ------------------------------

FILE = "models/data.pth"
model_data = torch.load(FILE, map_location=device)

input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']

nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_state)
nlp_model.eval()

# ------------------------------
# Load Data
# ------------------------------

diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].str.lower().str.strip()

disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].str.lower().str.strip()

symptom_severity = pd.read_csv("data/Symptom-severity.csv")

symptom_severity = symptom_severity.apply(
    lambda col: col.map(lambda s: s.lower().strip().replace(" ", "") if isinstance(s, str) else s)
)

# ------------------------------
# Load Symptoms and Prediction Model
# ------------------------------

with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)

with open('models/fitted_model.pickle2', 'rb') as modelFile:
    prediction_model = pickle.load(modelFile)

user_symptoms = set()

# ------------------------------
# Flask App
# ------------------------------

app = Flask(__name__)

# ------------------------------
# Health Check Endpoint
# ------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


# ------------------------------
# NLP Symptom Detection
# ------------------------------

def get_symptom(sentence):

    sentence = nltk.word_tokenize(sentence)

    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    with torch.no_grad():
        output = nlp_model(X)

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    return tag, prob


# ------------------------------
# Homepage
# ------------------------------

@app.route('/')
def index():

    user_symptoms.clear()

    data = []

    with open("static/assets/files/ds_symptoms.txt", "r") as file:
        all_symptoms = file.readlines()

    for s in all_symptoms:
        data.append(
            s.replace("'", "")
            .replace("_", " ")
            .replace(",\n", "")
        )

    data = json.dumps(data)

    return render_template('index.html', data=data)


# ------------------------------
# Symptom Prediction API
# ------------------------------

@app.route('/symptom', methods=['POST'])
def predict_symptom():

    sentence = request.json['sentence']

    # ------------------------------
    # User finished entering symptoms
    # ------------------------------

    if sentence.replace(".", "").replace("!", "").lower().strip() == "done":

        if not user_symptoms:

            response_sentence = random.choice([
                "I can't know what disease you may have if you don't enter any symptoms :)",
                "Meddy can't know the disease if there are no symptoms...",
                "You first have to enter some symptoms!"
            ])

        else:

            x_test = []

            for each in symptoms_list:
                if each in user_symptoms:
                    x_test.append(1)
                else:
                    x_test.append(0)

            x_test = np.asarray(x_test)

            disease = prediction_model.predict(
                x_test.reshape(1, -1)
            )[0]

            disease_clean = disease.strip().lower()

            description = diseases_description.loc[
                diseases_description['Disease'] == disease_clean,
                'Description'
            ].iloc[0]

            precaution = disease_precaution[
                disease_precaution['Disease'] == disease_clean
            ]

            precautions = (
                "Precautions: "
                + precaution.Precaution_1.iloc[0] + ", "
                + precaution.Precaution_2.iloc[0] + ", "
                + precaution.Precaution_3.iloc[0] + ", "
                + precaution.Precaution_4.iloc[0]
            )

            response_sentence = (
                f"It looks to me like you have {disease}. "
                "<br><br>"
                f"<i>Description: {description}</i>"
                "<br><br>"
                f"<b>{precautions}</b>"
            )

            # ------------------------------
            # Severity Calculation
            # ------------------------------

            severity = []

            for each in user_symptoms:

                weight = symptom_severity.loc[
                    symptom_severity['Symptom'] == each.lower().strip().replace(" ", ""),
                    'weight'
                ].iloc[0]

                severity.append(weight)

            if np.mean(severity) > 4 or np.max(severity) > 5:

                response_sentence += (
                    "<br><br>"
                    "Considering your symptoms are severe, "
                    "and Meddy isn't a real doctor, "
                    "you should consider talking to one."
                )

            user_symptoms.clear()

    # ------------------------------
    # User entering symptoms
    # ------------------------------

    else:

        symptom, prob = get_symptom(sentence)

        if prob > 0.5:

            response_sentence = (
                f"Hmm, I'm {(prob * 100):.2f}% sure "
                f"this is {symptom}."
            )

            user_symptoms.add(symptom)

        else:

            response_sentence = (
                "I'm sorry, but I don't understand you."
            )

    return jsonify(response_sentence.replace("_", " "))


# ------------------------------
# Local Development Only
# ------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)