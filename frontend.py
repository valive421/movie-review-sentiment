import streamlit as st
import requests

st.title("Sentiment Analysis of movie Reviews")
user_input = st.text_area(
    # The 'label' is the instruction text that appears above the text box.
    label="Enter the review text below:",
    # The 'placeholder' is the example text that appears inside the box.
    placeholder="This movie was fantastic and the acting was superb! I would highly recommend it to everyone.",
    # We can also set a default height for the text box.
    height=150
)
analyze_button = st.button("Analyze Sentiment")
if analyze_button:
    if user_input.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        # Define the API endpoint URL.
        API_URL = 'http://127.0.0.1:5000/predict'
        # Prepare the data to be sent in the POST request.
        data = {
            "review": user_input
        }
        # Send the POST request to the API.
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            sentiment = result['prediction']
            if sentiment.lower() == 'negative':
                st.error(f"Sentiment: {sentiment.capitalize()}")
            else:
                st.success(f"Sentiment: {sentiment.capitalize()}")
        else:
            st.error("Error occurred while analyzing sentiment.")