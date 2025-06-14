import os
import pickle
import numpy as np
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec

# Paths
DATA_PATH = "project2/data/IMDB Dataset.csv"
MODEL_DIR = "project2/models/"

# Ensure model dir exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_imdb_dataset():
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

def document_vector(doc, model):
    """Create document vector by averaging Word2Vec word vectors."""
    words = doc.split()
    word_vecs = []
    for w in words:
        if w in model.wv:
            word_vecs.append(model.wv[w])
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

def index(request):
    message = None
    selected_rep = None

    if request.method == "POST":
        rep_type = request.POST.get("representation")
        action = request.POST.get("action")
        selected_rep = rep_type

        X_train, X_test, y_train, y_test = load_imdb_dataset()

        if action == "train":
            # Train model depending on representation type
            if rep_type == "tfidf":
                vectorizer = TfidfVectorizer(max_features=10000)
                X_train_vec = vectorizer.fit_transform(X_train)
                clf = LogisticRegression(max_iter=5000)
                clf.fit(X_train_vec, y_train)

                X_test_vec = vectorizer.transform(X_test)
                y_pred = clf.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)

                # Save vectorizer and classifier
                with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
                    pickle.dump(vectorizer, f)
                with open(os.path.join(MODEL_DIR, "tfidf_classifier.pkl"), "wb") as f:
                    pickle.dump(clf, f)

                message = f"Trained TF-IDF model with accuracy: {acc:.4f}"

            elif rep_type == "bow":
                vectorizer = CountVectorizer(max_features=10000)
                X_train_vec = vectorizer.fit_transform(X_train)
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_vec, y_train)

                X_test_vec = vectorizer.transform(X_test)
                y_pred = clf.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)

                # Save vectorizer and classifier
                with open(os.path.join(MODEL_DIR, "bow_vectorizer.pkl"), "wb") as f:
                    pickle.dump(vectorizer, f)
                with open(os.path.join(MODEL_DIR, "bow_classifier.pkl"), "wb") as f:
                    pickle.dump(clf, f)

                message = f"Trained Bag of Words model with accuracy: {acc:.4f}"

            elif rep_type == "word2vec":
                # Train Word2Vec on training data only
                sentences = [text.split() for text in X_train]
                w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4, epochs=10)

                # Create document vectors
                X_train_vec = np.array([document_vector(text, w2v_model) for text in X_train])
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train_vec, y_train)

                X_test_vec = np.array([document_vector(text, w2v_model) for text in X_test])
                y_pred = clf.predict(X_test_vec)
                acc = accuracy_score(y_test, y_pred)

                # Save Word2Vec model and classifier
                w2v_model.save(os.path.join(MODEL_DIR, "word2vec_model.bin"))
                with open(os.path.join(MODEL_DIR, "word2vec_classifier.pkl"), "wb") as f:
                    pickle.dump(clf, f)

                message = f"Trained Word2Vec model with accuracy: {acc:.4f}"

            elif rep_type == "lda":
                # LDA requires CountVectorizer first
                count_vectorizer = CountVectorizer(max_features=5000)
                X_train_counts = count_vectorizer.fit_transform(X_train)

                lda = LatentDirichletAllocation(n_components=5, max_iter=5, random_state=42)
                X_train_lda = lda.fit_transform(X_train_counts)

                clf = LogisticRegression(max_iter=5000)
                clf.fit(X_train_lda, y_train)

                X_test_counts = count_vectorizer.transform(X_test)
                X_test_lda = lda.transform(X_test_counts)

                y_pred = clf.predict(X_test_lda)
                acc = accuracy_score(y_test, y_pred)

                # Save CountVectorizer and combined lda+clf as dict
                with open(os.path.join(MODEL_DIR, "lda_vectorizer.pkl"), "wb") as f:
                    pickle.dump(count_vectorizer, f)
                with open(os.path.join(MODEL_DIR, "lda_classifier.pkl"), "wb") as f:
                    pickle.dump({'lda': lda, 'clf': clf}, f)

                message = f"Trained LDA model with accuracy: {acc:.4f}"

            else:
                message = "Unknown representation type."

        elif action == "load":
            # Load pretrained model if exists and test accuracy
            try:
                if rep_type == "tfidf":
                    with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
                        vectorizer = pickle.load(f)
                    with open(os.path.join(MODEL_DIR, "tfidf_classifier.pkl"), "rb") as f:
                        clf = pickle.load(f)
                    X_test_vec = vectorizer.transform(X_test)
                    y_pred = clf.predict(X_test_vec)
                    acc = accuracy_score(y_test, y_pred)
                    message = f"Loaded pretrained TF-IDF model with test accuracy: {acc:.4f}"

                elif rep_type == "bow":
                    with open(os.path.join(MODEL_DIR, "bow_vectorizer.pkl"), "rb") as f:
                        vectorizer = pickle.load(f)
                    with open(os.path.join(MODEL_DIR, "bow_classifier.pkl"), "rb") as f:
                        clf = pickle.load(f)
                    X_test_vec = vectorizer.transform(X_test)
                    y_pred = clf.predict(X_test_vec)
                    acc = accuracy_score(y_test, y_pred)
                    message = f"Loaded pretrained Bag of Words model with test accuracy: {acc:.4f}"

                elif rep_type == "word2vec":
                    w2v_model = Word2Vec.load(os.path.join(MODEL_DIR, "word2vec_model.bin"))
                    with open(os.path.join(MODEL_DIR, "word2vec_classifier.pkl"), "rb") as f:
                        clf = pickle.load(f)
                    X_test_vec = np.array([document_vector(text, w2v_model) for text in X_test])
                    y_pred = clf.predict(X_test_vec)
                    acc = accuracy_score(y_test, y_pred)
                    message = f"Loaded pretrained Word2Vec model with test accuracy: {acc:.4f}"

                elif rep_type == "lda":
                    with open(os.path.join(MODEL_DIR, "lda_vectorizer.pkl"), "rb") as f:
                        count_vectorizer = pickle.load(f)
                    with open(os.path.join(MODEL_DIR, "lda_classifier.pkl"), "rb") as f:
                        classifier_dict = pickle.load(f)
                    lda = classifier_dict['lda']
                    clf = classifier_dict['clf']
                    X_test_counts = count_vectorizer.transform(X_test)
                    X_test_lda = lda.transform(X_test_counts)
                    y_pred = clf.predict(X_test_lda)
                    acc = accuracy_score(y_test, y_pred)
                    message = f"Loaded pretrained LDA model with test accuracy: {acc:.4f}"

                else:
                    message = "Unknown representation type."

            except FileNotFoundError:
                message = f"No pretrained model found for {rep_type}."
            except Exception as e:
                message = f"Error loading models: {str(e)}"

        else:
            message = "Invalid action."

    context = {
        "message": message,
        "selected_rep": selected_rep,
    }
    return render(request, "project2/project2_home.html", context)
