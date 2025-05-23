import streamlit as st
import pickle
import os
import gdown
import helper

MODEL_URL = "https://drive.google.com/uc?id=1jYJFaoQ5LEt4TOCX6FFIq0XvAy1TDbID"
MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return pickle.load(open(MODEL_PATH, 'rb'))

@st.cache_resource
def load_vectorizer():
    assert os.path.exists("cv.pkl"), "cv.pkl not found in repo."
    return pickle.load(open("cv.pkl", "rb"))

@st.cache_resource
def load_stopwords():
    assert os.path.exists("stopwords.pkl"), "stopwords.pkl not found in repo."
    return pickle.load(open("stopwords.pkl", "rb"))

# Inject dependencies into helper
model = load_model()
helper.cv = load_vectorizer()
helper.STOP_WORDS = load_stopwords()

# UI
st.title('🔍 DupDetect - Question Pair Duplicate Checker')

q1 = st.text_input('Enter question 1')
q2 = st.text_input('Enter question 2')

if st.button('Detect Duplicate'):
    if not q1.strip() or not q2.strip():
        st.warning("Both questions must be filled.")
    else:
        query = helper.query_point_creator(q1, q2)
        result = model.predict(query)[0]
        if result:
            st.success("✅ Duplicate")
        else:
            st.info("❌ Not Duplicate")
