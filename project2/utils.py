# project2/utils.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'IMDB Dataset.csv')

def load_imdb_dataset():
    """Load and preprocess the IMDB dataset"""
    df = pd.read_csv(DATA_PATH)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df['review'], df['sentiment']

def document_vector(doc, model):
    """Create document vector by averaging Word2Vec word vectors"""
    words = doc.split()
    word_vecs = [model.wv[w] for w in words if w in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

def initialize_vectorizer(model_type, X_train):
    """Initialize and fit the appropriate vectorizer"""
    if model_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X_train)
        return vectorizer, X_vec
    elif model_type == 'bow':
        vectorizer = CountVectorizer(max_features=10000, ngram_range=(1, 2))
        X_vec = vectorizer.fit_transform(X_train)
        return vectorizer, X_vec
    elif model_type == 'word2vec':
        sentences = [text.split() for text in X_train]
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)
        X_vec = np.array([document_vector(text, model) for text in X_train])
        return model, X_vec
    elif model_type == 'lda':
        count_vectorizer = CountVectorizer(max_features=5000)
        X_counts = count_vectorizer.fit_transform(X_train)
        lda = LatentDirichletAllocation(n_components=50, max_iter=5, random_state=42)
        X_vec = lda.fit_transform(X_counts)
        return {'count_vectorizer': count_vectorizer, 'lda': lda}, X_vec
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }

def get_active_learning_stats(clf, X_labeled, y_labeled, X_pool):
    """Calculate statistics for active learning"""
    if len(X_pool) > 0:
        probas = clf.predict_proba(X_pool)
        uncertainty = 1 - np.max(probas, axis=1).mean()
    else:
        uncertainty = 0
        
    return {
        'labeled_count': len(X_labeled),
        'pool_count': len(X_pool),
        'uncertainty': uncertainty
    }

def generate_filename(model_type, accuracy=None):
    """Generate a standardized filename for models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if accuracy is not None:
        return f"{model_type}_{accuracy:.4f}_{timestamp}.pkl"
    return f"{model_type}_{timestamp}.pkl"

def transform_text(text, vectorizer, model_type):
    """Transform text samples based on model type"""
    if isinstance(text, (list, np.ndarray, pd.Series)):
        if model_type == 'tfidf' or model_type == 'bow':
            return vectorizer.transform(text)
        elif model_type == 'word2vec':
            return np.array([document_vector(t, vectorizer) for t in text])
        elif model_type == 'lda':
            return vectorizer['lda'].transform(vectorizer['count_vectorizer'].transform(text))
    else:
        if model_type == 'tfidf' or model_type == 'bow':
            return vectorizer.transform([text]).toarray()[0]
        elif model_type == 'word2vec':
            return document_vector(text, vectorizer)
        elif model_type == 'lda':
            return vectorizer['lda'].transform(vectorizer['count_vectorizer'].transform([text]))[0]
    
    raise ValueError(f"Unknown model type: {model_type}")

def get_model_path(filename):
    """Get full path to a model file"""
    return os.path.join(MODEL_DIR, filename)