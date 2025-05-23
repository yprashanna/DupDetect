import streamlit as st
import helper
import pickle
import gdown
import os

def fetch_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        file_id = "1jYJFaoQ5LEt4TOCX6FFIq0XvAy1TDbID"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return pickle.load(open(model_path, 'rb'))

# Load model
model = fetch_model()

st.header('DupDetect')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Find'):
    query = helper.query_point_creator(q1, q2)
    result = model.predict(query)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')
