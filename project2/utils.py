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

# LDA Performance Settings - Optimized for PythonAnywhere
LDA_FAST_MODE = True  # Set to False for higher accuracy, True for faster training
LDA_SAMPLE_SIZE = 5000 if LDA_FAST_MODE else None   # Smaller dataset for faster training (reduced from 10000)
LDA_MAX_FEATURES = 500 if LDA_FAST_MODE else 2000   # Vocabulary size (reduced from 800)
LDA_N_COMPONENTS = 10 if LDA_FAST_MODE else 30      # Number of topics (reduced from 15)
LDA_MAX_ITER = 2 if LDA_FAST_MODE else 5            # Training iterations

# Active Learning Performance Settings
ACTIVE_LEARNING_SAMPLE_SIZE = 8000  # Reduced dataset size for active learning
INITIAL_LABELED_SIZE = 50            # Reduced from default 100

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'IMDB Dataset.csv')

def load_imdb_dataset(sample_size=None):
    """Load and preprocess the IMDB dataset
    
    Args:
        sample_size: If provided, randomly sample this many records for faster training
    """
    df = pd.read_csv(DATA_PATH)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # For LDA baseline training, we can use a smaller sample to speed up development
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
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
        # Optimized CountVectorizer with configurable parameters
        count_vectorizer = CountVectorizer(
            max_features=LDA_MAX_FEATURES,  # Configurable vocabulary size
            min_df=3,          # Remove very rare words
            max_df=0.85,       # Remove very common words  
            stop_words='english',  # Remove stop words
            ngram_range=(1, 1)     # Only unigrams for LDA (faster)
        )
        X_counts = count_vectorizer.fit_transform(X_train)
        
        # Highly optimized LDA parameters
        lda = LatentDirichletAllocation(
            n_components=LDA_N_COMPONENTS,  # Configurable number of topics
            max_iter=LDA_MAX_ITER,          # Configurable iterations
            learning_method='online',       # Faster online learning
            batch_size=256,                 # Larger batches for efficiency
            n_jobs=-1,                     # Use all CPU cores
            random_state=42,
            evaluate_every=1,              # Evaluate less frequently
            perp_tol=2e-2,                # More tolerant convergence (faster)
            learning_offset=20.,           # Faster initial learning
            learning_decay=0.6             # Faster learning decay
        )
        
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