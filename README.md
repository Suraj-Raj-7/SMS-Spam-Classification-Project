This project is an SMS Spam Classifier built with Python that uses a Multinomial Naive Bayes classifier to distinguish between spam and ham messages.

The project includes the following files:

README.md: A detailed description of the project, including the technology used, the steps for text preprocessing (removing punctuation, lowercasing, stopword filtering, and stemming), and instructions for setting up and running the application.

app.py: A Flask web application that serves the front-end and provides a /predict API endpoint for classifying SMS messages. It loads a pre-trained model to make predictions and returns the result as a JSON object.

create_model.py: A Python script to train the classification model. It reads a spam.csv dataset, preprocesses the text, creates a TfidfVectorizer to convert text into numerical features, trains a Multinomial Naive Bayes classifier, and saves the entire pipeline to a file named spam_classifier_model.pkl.

requirements.txt: Lists the Python dependencies required to run the project, including Flask, pandas, scikit-learn, nltk, and numpy.

static/css/style.css: A CSS file that styles the web application's user interface, including the layout, typography, and button appearance.

static/js/script.js: A JavaScript file that handles the client-side logic. It listens for a button click, sends the user's message to the /predict endpoint in app.py, and dynamically updates the webpage with the returned prediction.

spam.csv: The dataset of SMS messages used for training the model, with messages labeled as either "ham" (legitimate) or "spam".

spam_classifier_model.pkl: The saved, pre-trained machine learning model pipeline, which includes the vectorizer and the classifier. This file is loaded by app.py to make predictions without needing to retrain the model.

Activate the virtual environment:
venv\Scripts\activate

Run the script:
python app.py


