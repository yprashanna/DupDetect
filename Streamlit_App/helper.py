import re
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
import numpy as np

# These will be injected from app.py
cv = None
STOP_WORDS = None

def test_common_words(q1, q2):
    w1 = set(map(str.lower, q1.strip().split()))
    w2 = set(map(str.lower, q2.strip().split()))
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(map(str.lower, q1.strip().split()))
    w2 = set(map(str.lower, q2.strip().split()))
    return len(w1) + len(w2)

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    global STOP_WORDS

    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features

def test_fetch_fuzzy_features(q1, q2):
    fuzzy_features = [0.0] * 4
    fuzzy_features[0] = fuzz.QRatio(q1, q2)
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)
    return fuzzy_features

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ').replace('₹', ' rupee ')
    q = q.replace('€', ' euro ').replace('@', ' at ').replace('[math]', '')

    q = q.replace(',000,000,000', 'b').replace(',000,000', 'm').replace(',000', 'k')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "can't've": "can not have",
        "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "he's": "he is", "i'm": "i am", "isn't": "is not", "it's": "it is", "let's": "let us",
        "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they're": "they are", "wasn't": "was not", "we're": "we are", "weren't": "were not",
        "what's": "what is", "won't": "will not", "wouldn't": "would not", "you'd": "you would",
        "you'll": "you will", "you're": "you are"
    }

    q = ' '.join([contractions.get(w, w) for w in q.split()])
    q = q.replace("'ve", " have").replace("n't", " not").replace("'re", " are").replace("'ll", " will")

    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()

    return q

def query_point_creator(q1, q2):
    global cv
    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split()))
    input_query.append(len(q2.split()))
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    input_query.extend(test_fetch_token_features(q1, q2))
    input_query.extend(test_fetch_length_features(q1, q2))
    input_query.extend(test_fetch_fuzzy_features(q1, q2))

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, -1), q1_bow, q2_bow))
