

# Hackify Healthcare Platform

![image](https://github.com/user-attachments/assets/a1c00106-c4eb-4bba-8ed6-12de52d8f2ba)


A comprehensive healthcare platform designed to analyze food calories, predict diseases based on symptoms, and forecast medication side effects using patient data. This platform was developed during the IEDC Hackify Hackathon at MACE. Integrating machine learning with AI technologies, it provides real-time solutions for health-related issues.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Setup and Installation](#setup-and-installation)
5. [How to Run](#how-to-run)
6. [Food Calorie Analysis Pipeline](#food-calorie-analysis-pipeline)
7. [Disease Prediction and Health Chatbot](#disease-prediction-and-health-chatbot)
8. [Side Effects Prediction](#side-effects-prediction)
9. [Project Structure](#project-structure)
10. [Contributors](#contributors)

## Introduction

This platform utilizes AI, including YOLOv3 and OpenAI's GPT models, to deliver three core functionalities:

1. **Food Calorie Analysis**: Identifies food items in videos and estimates their calorie content.
2. **Personalized Disease Prediction**: Analyzes symptoms to predict possible diseases and offers advice.
3. **Genomic-Based Side Effect Prediction**: Predicts side effects of medications based on genomic data and patient details.

## Features

- **Food Calorie Analysis**:
  - Detects and annotates food items in real-time video using YOLOv3.
  - Estimates the calorie content of detected foods.
  
- **Personalized Disease Prediction**:
  - Analyzes symptoms using GPT-3.5 to predict diseases.
  - Provides personalized health advice based on disease predictions.
  
- **Side Effects Prediction**:
  - Predicts medication side effects using patient information and genomic data.
  - Offers insights into possible drug reactions and severity.

## Technologies Used

- **YOLOv3**: Object detection for real-time food analysis.
- **OpenAI GPT-3.5**: Natural language processing for disease prediction.
- **TensorFlow/Keras**: For model development and deployment.
- **OpenCV**: For video frame processing and food detection.
- **Random Forest Classifier**: For genomic-based side effect prediction.
- **Flask**: Web framework for deploying APIs.
- **SQLite**: For patient data storage.

## Setup and Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- OpenCV
- TensorFlow
- Keras
- Flask
- Supervision
- OpenAI Library
- scikit-learn (for Random Forest)

### Install Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

This will install all required packages such as OpenCV, TensorFlow, Flask, OpenAI, etc.

## How to Run

### 1. Food Calorie Analysis

Use the following script to analyze food items in a video:

```python
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
import cv2
import supervision as sv

annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_inference(predictions)
    image = annotator.annotate(scene=video_frame.image.copy(), detections=detections, labels=labels)
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    model_id="food_calorie/2",
    video_reference=r"C:\path\to\your\video.mp4",
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()
```

### 2. Disease Prediction and Health Chatbot

Use GPT-3.5 Turbo to predict diseases based on symptoms:

```python
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(model=model, messages=messages, temperature=0)
    return response.choices[0].message.content

def predict_disease(symptoms):
    prompt = "I am experiencing " + ", ".join(symptoms) + ". What could it be?"
    predicted_disease = get_completion(prompt)
    return predicted_disease

# Example usage
symptoms = ["fever", "cough", "sore throat"]
disease = predict_disease(symptoms)
print(f"Predicted disease: {disease}")
```

### 3. Side Effects Prediction

Use Random Forest to predict side effects from drug usage and patient details:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv

def load_drug_data(file_path):
    drug_data = []
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            drug_data.append(row)
    return drug_data

# Train the model and make predictions
drug_data = load_drug_data("side_effect_genotype.csv")
X, y, drug_label_encoder, genotype_label_encoder = preprocess_drug_data(drug_data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Example usage
prediction = predict_side_effects(rf_model, age=30, gender="M", height=175, weight=70, drug_name="Aspirin", genotype="AA")
print(f"Predicted side effects: {prediction[0]}")
```

## Project Structure

```
Hackify-Healthcare-Platform/
│
├── food_calorie_analysis/     
│   └── inference.py
│
├── disease_prediction/          
│   └── disease_predictor.py
│
├── side_effect_prediction/      
│   └── side_effect_predictor.py
│
├── web/                         
│   └── app.py
│
├── requirements.txt            
├── README.md                     
└── side_effect_genotype.csv       
```

## Contributors

- **Ashitha Pallath** -  Developer
- **Hira Mohammed** - Developer
- **Omal S** - Developer
- **Mariya Benny** - Developer

