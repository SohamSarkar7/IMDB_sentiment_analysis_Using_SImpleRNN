import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


words_hist = imdb.get_word_index()

reverse = {value: key for key,value in words_hist
           .items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encode_review):
    return " ".join([reverse.get(i-3,'?') for i in encode_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [words_hist.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    pre_processed = preprocess_text(review)

    prediction = model.predict(pre_processed)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment,prediction[0][0]


if __name__=='__main__':



    st.title("IMDB MOVIE Review Sentiment Analysis")
    st.write("Enter Your review and classify it as positive or negetive")

    user_input = st.text_area('Movie Review')

    if st.button('Classify'):

        preprocess_input = preprocess_text(user_input)

        prediction = model.predict(preprocess_input)

        sentiment = "positive" if prediction[0][0] > 0.6 else "Negative"

        st.write(f'Sentiment {sentiment}')
        st.write(f'Prediction score {prediction[0][0]}')
    else:
        st.write('Please enter a movie review')
