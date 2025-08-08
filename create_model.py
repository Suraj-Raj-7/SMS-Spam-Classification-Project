# create_model.py

# Step 1: Importing Libraries
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import pickle
import os

# Download stopwords from NLTK (This only needs to be run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


# Step 2: Load and Explore the Dataset
# Load the dataset
# You will need to make sure 'spam.csv' is in the same directory as this script.
df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the necessary columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Data Preprocessing
# Text Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]

    # Join the words back into a single string
    return ' '.join(words)

# Apply the preprocessing to all messages
df['message'] = df['message'].apply(preprocess_text)


# Step 4: Convert Text to Numerical Data
# Split the dataset into features (X) and labels (y)
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build and Train the Model
# Create a pipeline to automate the TF-IDF and classification process
# This is an important step to ensure the same transformations are applied to new data
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=3000)),
    ('classifier', MultinomialNB())
])

# Train the pipeline on the entire training dataset
pipeline.fit(X_train, y_train)

# Make predictions and evaluate performance (optional, but good for verification)
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.4f}')

# Step 6: Save the Trained Model to a PKL File
# This is the crucial part for your web application.
# The entire pipeline object is saved to a file named 'spam_classifier_model.pkl'.
# The 'wb' mode opens the file in binary write mode.
try:
    with open('spam_classifier_model.pkl', 'wb') as file:
        pickle.dump(pipeline, file)
    print("Successfully saved the model pipeline to 'spam_classifier_model.pkl'")
except Exception as e:
    print(f"Error saving the model: {e}")

# This part is just for testing the saved model
if os.path.exists('spam_classifier_model.pkl'):
    with open('spam_classifier_model.pkl', 'rb') as file:
        loaded_pipeline = pickle.load(file)
        
    def predict_with_loaded_model(message):
        prediction = loaded_pipeline.predict([message])[0]
        return 'Spam' if prediction == 1 else 'Ham'

    test_message1 = "Free entry in 2 a wkly comp to win FA Cup final tkts"
    test_message2 = "Hey there darling, how are you today?"
    
    print('\nTesting predictions with the saved model:')
    print(f"Message: '{test_message1}' -> Prediction: {predict_with_loaded_model(test_message1)}")
    print(f"Message: '{test_message2}' -> Prediction: {predict_with_loaded_model(test_message2)}")