# -*- coding: utf-8 -*-
"""
Question-Answering Model

This script processes text data for a question-answering model using NLP techniques.
It includes preprocessing, feature engineering, and model training with evaluation.

Automatically generated by Colab.
Original file is located at: https://colab.research.google.com/drive/1w1IQ6CvG5IkHEKPphvXx8mOnXdKVScHt
"""

# Install necessary packages
!pip install transformers

# Import libraries
import transformers
import torch
import numpy as np
import pandas as pd
import random
import string
import pickle
import datetime
import nltk
import tensorflow as tf
import tensorflow_hub as hub

from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
from scipy import stats
from scipy.stats import rankdata

# Settings
DATA_DIR = '../input/google-quest-challenge/'
METAS_DIR = ''
SUB_DIR = ''
RANDOM_STATE = 42
SEED = 42
LIMIT_CHAR = 5000
LIMIT_WORD = 25000

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)

# Download stopwords
nltk.download('stopwords')
ENG_STOPWORDS = set(stopwords.words("english"))

def fetch_vectors(string_list, batch_size=64):
    """Fetch embeddings for a list of strings using DistilBERT."""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in chunks(string_list, batch_size):
        tokenized = [tokenizer.encode(" ".join(x.strip().split()[:300]), add_special_tokens=True)[:512] for x in data]
        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    return np.vstack(fin_features)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def spearman_corr(y_true, y_pred):
    """Compute Spearman correlation coefficient."""
    if np.ndim(y_pred) == 2:
        return np.mean([stats.spearmanr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
    return stats.spearmanr(y_true, y_pred)[0]

custom_scorer = make_scorer(spearman_corr, greater_is_better=True)

def preprocess_data():
    """Load and preprocess the data."""
    xtrain = pd.read_csv(DATA_DIR + 'train.csv')
    xtest = pd.read_csv(DATA_DIR + 'test.csv')
    
    for colname in ['question_title', 'question_body', 'answer']:
        xtrain[colname + '_word_len'] = xtrain[colname].str.split().str.len()
        xtest[colname + '_word_len'] = xtest[colname].str.split().str.len()

    for colname in ['question', 'answer']:
        xtrain['is_' + colname + '_no_name_user'] = xtrain[colname + '_user_name'].str.contains('^user\d+$') + 0
        xtest['is_' + colname + '_no_name_user'] = xtest[colname + '_user_name'].str.contains('^user\d+$') + 0

    xtrain['answer_div'] = xtrain['answer'].apply(lambda s: len(set(s.split())) / len(s.split()))
    xtest['answer_div'] = xtest['answer'].apply(lambda s: len(set(s.split())) / len(s.split()))

    for df in [xtrain, xtest]:
        df['domcom'] = df['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))
        df['dom_cnt'] = df['domcom'].apply(lambda s: len(s))
        df['domcom'] = df['domcom'].apply(lambda s: s + ['none', 'none'])

        for ii in range(0, 4):
            df['dom_' + str(ii)] = df['domcom'].apply(lambda s: s[ii])

    xtrain.drop(['domcom'], axis=1, inplace=True)
    xtest.drop(['domcom'], axis=1, inplace=True)

    for df in [xtrain, xtest]:
        df['q_words'] = df['question_body'].apply(lambda s: [f for f in s.split() if f not in ENG_STOPWORDS])
        df['a_words'] = df['answer'].apply(lambda s: [f for f in s.split() if f not in ENG_STOPWORDS])
        df['qa_word_overlap'] = df.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis=1)
        df['qa_word_overlap_norm1'] = df.apply(lambda s: s['qa_word_overlap'] / (1 + len(s['a_words'])), axis=1)
        df['qa_word_overlap_norm2'] = df.apply(lambda s: s['qa_word_overlap'] / (1 + len(s['q_words'])), axis=1)
        df.drop(['q_words', 'a_words'], axis=1, inplace=True)

        df["question_title_num_chars"] = df["question_title"].apply(lambda x: len(str(x)))
        df["question_body_num_chars"] = df["question_body"].apply(lambda x: len(str(x)))
        df["answer_num_chars"] = df["answer"].apply(lambda x: len(str(x)))

        df["question_title_num_stopwords"] = df["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in ENG_STOPWORDS]))
        df["question_body_num_stopwords"] = df["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in ENG_STOPWORDS]))
        df["answer_num_stopwords"] = df["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in ENG_STOPWORDS]))

        df["question_title_num_punctuations"] = df['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        df["question_body_num_punctuations"] = df['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
        df["answer_num_punctuations"] = df['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

        df["question_title_num_words_upper"] = df["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        df["question_body_num_words_upper"] = df["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
        df["answer_num_words_upper"] = df["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

    return xtrain, xtest

def compute_embeddings(xtrain, xtest):
    """Compute sentence embeddings using Universal Sentence Encoder."""
    module_url = "../input/universalsentenceencoderlarge4/"
    embed = hub.load(module_url)

    embeddings_train = {}
    embeddings_test = {}
    for text in ['question_title', 'question_body', 'answer']:
        train_text = xtrain[text].str.replace('?', '.').str.replace('!', '.').tolist()
        test_text = xtest[text].str.replace('?', '.').str.replace('!', '.').tolist()

        curr_train_emb = []
        curr_test_emb = []
        batch_size = 4

        for ind in range(0, len(train_text), batch_size):
            curr_train_emb.append(embed(train_text[ind: ind + batch_size])["outputs"].numpy())
        for ind in range(0, len(test_text), batch_size):
            curr_test_emb.append(embed(test_text[ind: ind + batch_size])["outputs"].numpy())

        embeddings_train[text] = np.vstack(curr_train_emb)
        embeddings_test[text] = np.vstack(curr_test_emb)

    return embeddings_train, embeddings_test

def train_and_evaluate(xtrain, embeddings_train):
    """Train the model and evaluate performance."""
    feats = ['question_title_num_chars', 'question_body_num_chars', 'answer_num_chars',
             'question_title_num_stopwords', 'question_body_num_stopwords', 'answer_num_stopwords',
             'question_title_num_punctuations', 'question_body_num_punctuations', 'answer_num_punctuations',
             'question_title_num_words_upper', 'question_body_num_words_upper', 'answer_num_words_upper',
             'qa_word_overlap', 'qa_word_overlap_norm1', 'qa_word_overlap_norm2', 'dom_0', 'dom_1', 'dom_2', 'dom_3', 'dom_cnt']

    X_train = xtrain[feats].values
    y_train = xtrain['target'].values
    model = Ridge(alpha=1.0)

    model.fit(X_train, y_train)
    preds = model.predict(X_train)
    score = spearman_corr(y_train, preds)

    print(f"Training score: {score:.4f}")

def main():
    """Main function to run the preprocessing, embedding, and model training."""
    xtrain, xtest = preprocess_data()
    embeddings_train, embeddings_test = compute_embeddings(xtrain, xtest)
    train_and_evaluate(xtrain, embeddings_train)
    
if __name__ == "__main__":
    main()
