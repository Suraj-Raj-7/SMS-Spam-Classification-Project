import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    Main function to load the model and evaluate it.
    """
    # Define the path to the model file and the dataset
    model_path = 'spam_classifier_model.pkl'
    data_path = 'spam.csv'

    # Check if the required files exist
    if not os.path.exists(model_path):
        print(f"Error: The model file '{model_path}' was not found.")
        print("Please ensure 'spam_classifier_model.pkl' is in the same directory.")
        return

    if not os.path.exists(data_path):
        print(f"Error: The dataset file '{data_path}' was not found.")
        print("Please ensure 'spam.csv' is in the same directory.")
        return

    # Load the trained pipeline from the .pkl file
    try:
        with open(model_path, 'rb') as f:
            pipeline = pickle.load(f)
        print("Successfully loaded the model pipeline.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Load the dataset to get the test data
    # Use the 'usecols' parameter to handle extra columns in the CSV
    try:
        df = pd.read_csv(data_path, encoding='latin-1', usecols=['v1', 'v2'])
        df.columns = ['label', 'message']
    except ValueError:
        # Fallback for CSV files that might have different column names
        df = pd.read_csv(data_path, encoding='latin-1')
        df.columns = ['label', 'message', 'col3', 'col4', 'col5']
        df = df[['label', 'message']]

    # Map labels to numerical values
    y = df['label'].map({'ham': 0, 'spam': 1})
    X = df['message']

    # Split the data into training and testing sets with a fixed random_state
    # This must be the same split used in `create_model.py` for a fair evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use the loaded pipeline to make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # ========================================================================================================
    # Multi-line comment explaining the stats
    # ========================================================================================================
    """
    Here's a breakdown of the performance metrics you see in the classification report.

    Accuracy:
    This is the overall percentage of messages that the model classified correctly.
    An accuracy of 0.957 means the model was correct 95.7% of the time. However, this
    can be misleading if the data is unbalanced, which is common in spam classification
    (there are many more "ham" messages than "spam" messages).

    Precision:
    Precision tells you how many of the messages the model predicted as a certain class
    were actually correct. It answers the question: "Of all the messages I labeled as SPAM,
    how many were truly SPAM?"
    - A precision of 1.00 for spam (class 1) means that every single message the model
      flagged as spam was actually a spam message. There were no false positives.

    Recall:
    Recall tells you how many of the actual messages of a certain class the model was
    able to find. It answers the question: "Of all the messages that were truly SPAM,
    how many did I correctly identify?"
    - A recall of 0.68 for spam (class 1) means the model only caught 68% of the
      real spam messages. The other 32% of spam messages were incorrectly
      classified as "ham" (false negatives).

    F1-Score:
    The F1-score is a single metric that balances both precision and recall. It's a
    more useful measure than accuracy for imbalanced datasets. It's the harmonic mean
    of precision and recall.
    - An F1-score of 0.81 for spam (class 1) indicates that while the model
      is very precise (it doesn't make many mistakes when it labels something as spam),
      it has a hard time catching all of the spam messages.

    Support:
    This is simply the number of messages for each class in your test set.
    - There were 965 "ham" messages (class 0) and 150 "spam" messages (class 1) in your test data.
    """
    # ========================================================================================================

    # Print performance metrics
    print("\nModel Performance Metrics on the Test Set:")
    print("-" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

if __name__ == '__main__':
    main()
