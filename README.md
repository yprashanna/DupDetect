
# 🔍 DupDetect – Question Pair Duplicate Checker

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Available-brightgreen)](https://duplicatedetector.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DupDetect is a machine learning–powered web application that identifies whether two user-submitted questions are duplicates. Inspired by Quora's question pair challenge, this project combines NLP techniques, feature engineering, and classification models to deliver accurate and real-time predictions via an intuitive Streamlit interface.

---

## 🚀 Live Demo

👉 [Try the App](https://duplicatedetector.streamlit.app/)

---

## 📌 Features

- 🧠 **Advanced Feature Engineering**: Combines TF-IDF vectors, handcrafted semantic features, and fuzzy string similarity scores.
- ⚙️ **Custom Preprocessing Pipeline**: Includes text normalization, stopword filtering, and contraction handling for robust input cleaning.
- 📈 **Progressive Accuracy Improvement**:
  - Bag of Words only: ~74%
  - + Basic Features: ~76%
  - + Advanced NLP Features: ~80%
- 🌐 **Deployed with Streamlit**: Fast, responsive, and publicly accessible UI hosted on Streamlit Cloud.
- 🧳 **Lightweight Model Loading**: Uses caching and Google Drive integration to efficiently load model assets on demand.

---

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: Scikit-learn, Pandas, NumPy, FuzzyWuzzy, Distance, BeautifulSoup
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Model Hosting**: Google Drive + gdown
- **Deployment**: [Streamlit Cloud](https://streamlit.io/cloud)

---

## 📂 Project Structure

```
yprashanna-dupdetect/
├── Added_Basic_Features.ipynb               # Feature-engineered model with basic linguistic features
├── cv.pkl                                   # Trained CountVectorizer / TF-IDF model
├── Initial_EDA.ipynb                        # Exploratory Data Analysis notebook
├── Only_BOW.ipynb                           # Baseline model using Bag of Words
├── Preprocessing_and_Advanced_Features.ipynb# Final model with advanced NLP features
├── stopwords.pkl                            # Preprocessed stopword list
├── Streamlit_App/                           # Streamlit web application
│   ├── app.py                               # Streamlit app logic and UI
│   ├── cv.pkl                               # Vectorizer for use in app
│   ├── helper.py                            # Feature extraction and preprocessing
│   ├── requirements.txt                     # Python dependencies
│   └── stopwords.pkl                        # Stopwords used in app logic
└── .devcontainer/
    └── devcontainer.json                    # Optional dev environment config (VS Code Remote)

├── app.py                 # Streamlit frontend logic
├── helper.py              # Feature engineering & preprocessing logic
├── model.pkl              # Trained classification model
├── cv.pkl                 # CountVectorizer or TF-IDF vectorizer
├── stopwords.pkl          # Stopword list used in preprocessing
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yprashanna/DupDetect.git
cd DupDetect
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run app.py
```

> 🔁 Make sure `model.pkl`, `cv.pkl`, and `stopwords.pkl` are available or will be downloaded from the provided link on first run.

---

## 📊 Model and Feature Details

The feature vector used by the model includes:
- Length-based features (word count, character count)
- Common word/stopword/token ratios
- Fuzzy string similarity metrics (QRatio, Partial Ratio, Token Set/Sort)
- TF-IDF vectors of both questions

The final vector is fed to a `Multinomial Naive Bayes` classifier trained on labeled question pairs.

---

## 🧠 Inspiration

This project is inspired by Quora's duplicate question detection problem, aiming to explore how different NLP strategies can improve semantic similarity understanding.

---

## 🙌 Acknowledgements

- [Quora Question Pairs Dataset](https://www.kaggle.com/c/quora-question-pairs)
- [Streamlit](https://streamlit.io/)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)

---

## 💬 Contact

📧 **Prashanna Yadav** – [LinkedIn](https://www.linkedin.com/in/prashannaky/)  
📁 View the full project: [GitHub Repo](https://github.com/yprashanna/DupDetect)
