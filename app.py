from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

try:
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    # Print a success message to the console to confirm that the models loaded correctly.
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer file not found. Make sure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as app.py.")
    model = None
    vectorizer = None
except Exception as e:
    print(f"An error occurred while loading the model/vectorizer: {e}")
    model = None
    vectorizer = None


# --- ADD THE PREPROCESSING FUNCTION BELOW THE NLTK SETUP ---

def preprocess_text(text):
    """
    This function takes a raw text string and performs a series of cleaning
    and preprocessing steps, returning a clean string ready for TF-IDF vectorization.
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenize the text
    tokens = text.split()

    # 5. Remove stopwords and perform lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    # 6. Join tokens back into a single string
    return ' '.join(cleaned_tokens)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # A basic check to ensure we received data and it's not empty.
        if data is None:
            # Return a user-friendly error message and a 400 Bad Request status code.
            return jsonify({'error': 'Invalid input: No JSON data received or content-type is not application/json.'}), 400
    except Exception as e:
        # If any other error occurs during JSON parsing, we catch it here.
        return jsonify({'error': f'An error occurred while parsing JSON: {str(e)}'}), 400
    review_text = data.get('review')
    if not review_text:
        return jsonify({'error': 'Invalid input: No review text provided.'}), 400

    # Preprocess the review text
    processed_text = preprocess_text(review_text)
    print(processed_text)

    # Vectorize the processed text
    X = vectorizer.transform([processed_text])
    print(X)
    # Make a prediction
    prediction = model.predict(X)
    print(prediction)
    # Return the prediction result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
