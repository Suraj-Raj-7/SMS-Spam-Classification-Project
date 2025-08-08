# app.py

import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Define the path to the model file
# It is important to load the model once when the application starts
model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier_model.pkl')
try:
    with open(model_path, 'rb') as f:
        # Load the entire pipeline that was saved, including the vectorizer and classifier
        pipeline = pickle.load(f)
except FileNotFoundError:
    print(f"Error: The model file '{model_path}' was not found.")
    print("Please run the provided Python script to generate the .pkl file.")
    pipeline = None

@app.route('/')
def home():
    """
    This function is the main route for the web application. 
    It is responsible for rendering the 'index.html' file to the user's browser.
    It does not receive any input from the client directly.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function acts as the API endpoint for making predictions.
    
    1.  **Input:** It receives a JSON object via a POST request from the client-side JavaScript.
        The `script.js` file sends this request when the "Classify Message" button is clicked.
        The JSON object contains a key named 'message' with the user's input SMS text.

    2.  **Processing:** - It first checks if the model was loaded successfully.
        - It then extracts the 'message' string from the received JSON data.
        - The `pipeline` object, which contains both the `TfidfVectorizer` and the `MultinomialNB` classifier, takes the raw text message. The pipeline automatically preprocesses the message (lowercasing, punctuation removal, stemming, etc.), converts it into a numerical format, and then feeds it to the classifier to get a prediction (0 for Ham, 1 for Spam).

    3.  **Output:** It sends a JSON response back to the `script.js` file.
        - The JSON response contains a key named 'prediction' with a string value of either 'Spam' or 'Ham'.
        - `script.js` then uses this response to update the result section on the web page.
    """
    if not pipeline:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request: "message" field is missing'}), 400

    message_to_predict = data['message']
    
    # Use the loaded pipeline to make a prediction
    prediction = pipeline.predict([message_to_predict])
    
    result = 'Spam' if prediction[0] == 1 else 'Ham'
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Makes the server available locally for development
    app.run(debug=True)