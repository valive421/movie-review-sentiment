# Sentiment Analysis Web App

A complete end-to-end machine learning project for sentiment analysis of IMDB movie reviews. This project demonstrates data cleaning, feature engineering, classical ML, deep learning, and deployment via a Flask API and Streamlit frontend.

## Live link :
https://depseanmovie-reviews-3few29xxbtetyrtws6l7bj.streamlit.app/

## Features
- **Data Exploration & Cleaning**: Jupyter notebook for EDA and preprocessing
- **Classical ML Model**: Logistic Regression with TF-IDF features
- **Deep Learning Model**: LSTM using TensorFlow/Keras
- **REST API**: Flask app for model inference
- **Frontend**: Streamlit web app for user interaction
- **Automated Testing**: Python script to test the API


## DEMO


https://github.com/user-attachments/assets/3f5096ea-ebd0-4204-9a7b-c49c4192b19e





## Project Structure
```
project/
│   app.py                # Flask API backend
│   frontend.py           # Streamlit frontend
│   test_api.py           # API test script
│   logistic_regression_model.pkl  # Saved ML model
│   tfidf_vectorizer.pkl  # Saved vectorizer
│   requirements.txt      # Python dependencies
│   README.md             # Project documentation
│
├── data/
│   └── IMDB Dataset.csv  # Raw dataset
│
└── notebook/
    └── 01-data-exploration-and-cleaning.ipynb.ipynb  # Data cleaning & modeling notebook
```

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone https://github.com/valive421/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download NLTK data**
   In Python shell or notebook:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
5. **Run the Jupyter notebook**
   - Open `notebook/01-data-exploration-and-cleaning.ipynb.ipynb` and run all cells to train and save the models.

6. **Start the Flask API**
   ```bash
   python app.py
   ```

7. **Test the API**
   ```bash
   python test_api.py
   ```

8. **Run the Streamlit frontend**
   ```bash
   streamlit run frontend.py
   ```

## API Usage
- **Endpoint:** `POST /predict`
- **Request JSON:** `{ "review": "Your review text here" }`
- **Response JSON:** `{ "prediction": "positive" | "negative" }`

## Model Performance

### Logistic Regression (TF-IDF)
- **Test Accuracy:** ~88% (may vary depending on random seed and preprocessing)

### LSTM (Deep Learning)
- **Test Accuracy:** ~89% (may vary depending on random seed and preprocessing)

## Example
```python
import requests
response = requests.post('http://127.0.0.1:5000/predict', json={"review": "Great movie!"})
print(response.json())
```

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, nltk, Flask, joblib, streamlit, tensorflow, matplotlib, seaborn, requests

## Acknowledgements
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Scikit-learn](https://scikit-learn.org/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)


---

**⭐ If you found this project useful, please star the repo!**
