import streamlit as st
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib

# Load data
df = pd.read_csv('cleaned_news.csv')

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a string
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

# Preprocess text data
df['preprocessed_text'] = df['article_text'].map(preprocess_text)

# Streamlit UI
st.title('Text Category Prediction')

new_input = st.text_area('Enter text:', 'Type here...')
if st.button('Predict'):
    # Load the trained model and vectorizer
    kmeans_model = joblib.load('kmeans_model (2).pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Preprocess the new input
    processed_input = preprocess_text(new_input)

    # Vectorize the preprocessed input
    input_vectorized = vectorizer.transform([processed_input])

    # Predict the cluster
    cluster = kmeans_model.predict(input_vectorized)[0]

    # Assign category based on cluster
    if cluster == 0:
        category = 'Business'
    elif cluster == 1:
        category = 'Health'
    else:
        category = 'Sports'

    st.write('Predicted category:', category)

