import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os

MODEL_PATH = "project2/models/"
os.makedirs(MODEL_PATH, exist_ok=True)

def load_imdb_dataset():
    df = pd.read_csv("project2/data/IMDB Dataset.csv")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

def train_and_save_model():
    X_train, X_test, y_train, y_test = load_imdb_dataset()
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000)),
        ('clf', LogisticRegression(max_iter=5000))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    joblib.dump(pipeline, os.path.join(MODEL_PATH, "sentiment_model.pkl"))
    
    return acc

def load_model():
    return joblib.load(os.path.join(MODEL_PATH, "sentiment_model.pkl"))
