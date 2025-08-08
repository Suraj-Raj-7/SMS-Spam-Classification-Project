SMS Spam Classifier
This project is a Python-based SMS Spam Classifier that utilizes a machine learning model to accurately identify text messages as either "spam" or "ham" (legitimate). It's a full-stack application with a Flask backend and a simple web interface.

Technologies Used
Python: The core programming language for the backend logic and machine learning model.

Flask: A lightweight web framework used to build the web server and API endpoints for the classifier.

scikit-learn: A powerful machine learning library for Python. It is used to build the TfidfVectorizer for text feature extraction and the Multinomial Naive Bayes classifier for the core classification logic.

pandas: Utilized for data manipulation and analysis, specifically for reading and processing the spam.csv dataset.

NLTK (Natural Language Toolkit): A library for natural language processing, used for text preprocessing tasks such as stopword removal and stemming.

HTML, CSS, & JavaScript: The front-end is built with these web technologies to provide a simple and user-friendly interface for entering and classifying messages.

Project Structure
app.py: Contains the Flask application, which serves the index.html file and provides the /predict API endpoint to classify user-submitted messages.

create_model.py: A standalone script that loads the spam.csv dataset, preprocesses the data, trains the machine learning pipeline (vectorizer + classifier), and saves it as spam_classifier_model.pkl.

requirements.txt: Lists all the necessary Python libraries and their versions for dependency management.

spam.csv: The dataset used for training the model.

spam_classifier_model.pkl: The pre-trained machine learning model file that the Flask application uses for predictions.

templates/index.html: The HTML file for the web interface.

static/css/style.css: The CSS file for styling the web page.

static/js/script.js: The JavaScript file for handling front-end interactions and communicating with the Flask backend.

Activate the virtual environment:
venv\Scripts\activate

Run the script:
python app.py


