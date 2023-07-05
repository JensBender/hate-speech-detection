# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 08:39:20 2023

@author: Jens Bender
"""

from flask import Flask, render_template, request
import tensorflow as tf
import tensorflow_text  # prerequisite for using the BERT preprocessing layer
import numpy as np

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model("saved_models/model3")

# Define the route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Define the route for model prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve the input text from the HTML form
    input_text = request.form["input_text"]
    # Convert input text to a list
    input_data = [input_text]
    # Make prediction using the TensorFlow model
    prediction_prob = model.predict(input_data)[0][0]
    # Convert prediction probability to percent
    prediction_prob = np.round(prediction_prob * 100, 1)
    # Convert prediction probability to prediction text
    if prediction_prob >= 50:
        prediction = "Hate Speech"
    else:
        prediction = "No Hate Speech"
        # Invert the prediction probability
        prediction_prob = 100 - prediction_prob
    # Render the prediction, prediction probability and input text in the index.html template
    return render_template("index.html", prediction=prediction, prediction_prob=prediction_prob, text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
