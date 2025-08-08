This project is an SMS Spam Classifier built using Python, which reads a dataset of SMS messages (spam.csv), preprocesses the text by removing punctuation, lowercasing, stopword filtering, and stemming, then converts the messages into TF-IDF features to train a Multinomial Naive Bayes classifier that can accurately distinguish between spam and ham. The model achieves high accuracy and can be tested on custom input messages. All processing and training steps are handled inside the main Python script. For best practice, the project uses a virtual environment (venv) to isolate dependencies and supports saving the model and vectorizer for reuse.

Activate the virtual environment:
venv\Scripts\activate

Run the script:
python spam_classifier.py


