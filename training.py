# train.py
import os
import pandas as pd
import numpy as np
import pickle
import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

nltk.download('stopwords')

# ---- Config ----
DATA_PATH = "spam.csv"  # update path if needed
RNN_MODEL_PATH = "rnn_model.h5"
NB_PIPELINE_PATH = "nb_pipeline.pkl"
TOKENIZER_PATH = "tokenizer.pkl"

MAX_NUM_WORDS = 20000
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 8

# ---- Utilities ----
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# ---- Load dataset ----
df = pd.read_csv(DATA_PATH, encoding='latin-1')
# Dataset formats vary; handle common Kaggle `spam.csv` which has columns 'v1' (label) and 'v2' (text)
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1','v2']].rename(columns={'v1':'label','v2':'text'})
else:
    # fallback: look for common names
    df = df.rename(columns={df.columns[0]:'label', df.columns[1]:'text'})[['label','text']]

df['label_num'] = df['label'].map({'ham':0, 'spam':1})

# Clean
df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text'].values
y = df['label_num'].values

# ---- Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# ---- Train Naive Bayes (TF-IDF + MultinomialNB) ----
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

y_pred_nb = nb.predict(X_test_tfidf)
print("Naive Bayes accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Save the NB pipeline (we'll store tfidf and nb together)
nb_pipeline = {'tfidf': tfidf, 'nb': nb}
with open(NB_PIPELINE_PATH, 'wb') as f:
    pickle.dump(nb_pipeline, f)
print("Saved NB pipeline to", NB_PIPELINE_PATH)

# ---- Prepare sequences for RNN (Tokenizer + sequences) ----
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Compute NB probabilities for inputs (as scalar features)
X_train_nb_prob = nb.predict_proba(tfidf.transform(X_train))[:,1].reshape(-1,1)  # prob of spam
X_test_nb_prob = nb.predict_proba(tfidf.transform(X_test))[:,1].reshape(-1,1)

# ---- Build RNN model that takes sequence + NB_prob ----
# Sequence input
seq_input = Input(shape=(MAX_SEQ_LEN,), name='seq_input')
x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_SEQ_LEN)(seq_input)
x = LSTM(64, return_sequences=False)(x)
x = Dropout(0.4)(x)

# NB prob input
nb_input = Input(shape=(1,), name='nb_input')

# Concatenate
concat = Concatenate()([x, nb_input])
dense = Dense(64, activation='relu')(concat)
dense = Dropout(0.3)(dense)
out = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[seq_input, nb_input], outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---- Train RNN ----
es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(
    {'seq_input': X_train_pad, 'nb_input': X_train_nb_prob},
    y_train,
    validation_data=({'seq_input': X_test_pad, 'nb_input': X_test_nb_prob}, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es],
    verbose=1
)

# Evaluate
loss, acc = model.evaluate({'seq_input': X_test_pad, 'nb_input': X_test_nb_prob}, y_test, verbose=0)
print("RNN test accuracy:", acc)

# Save RNN model and tokenizer
model.save(RNN_MODEL_PATH)
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump({'tokenizer': tokenizer, 'max_seq_len': MAX_SEQ_LEN, 'max_words':MAX_NUM_WORDS}, f)

print("Saved RNN model to", RNN_MODEL_PATH)
print("Saved tokenizer to", TOKENIZER_PATH)
