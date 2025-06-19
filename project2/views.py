import os
import json
import pickle
import time
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec
from .model_utils import save_model, load_model, update_metadata, list_models, load_metadata
from .utils import (load_imdb_dataset, initialize_vectorizer, 
                   evaluate_model, generate_filename, transform_text, document_vector)

# Global session storage (in production, use Django's session framework or database)
ACTIVE_LEARNING_SESSIONS = {}

# Model directory configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def index(request):
    """Main dashboard view"""
    return render(request, 'project2/project2_home.html', {
        'models': list_models()
    })

@csrf_exempt
def train_model(request, model_type):
    """Train a new model of specified type"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        X, y = load_imdb_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        vectorizer, X_vec = initialize_vectorizer(model_type, X_train)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_vec, y_train)
        
        # Evaluate model
        X_test_vec = transform_text(X_test, vectorizer, model_type)
        accuracy = accuracy_score(y_test, clf.predict(X_test_vec))
        
        # Save model
        filename = generate_filename(model_type, accuracy)
        save_model(clf, filename)
        update_metadata(model_type, filename, accuracy)
        
        # Save vectorizer if needed
        if model_type in ['tfidf', 'bow', 'lda']:
            vec_filename = f"{model_type}_vectorizer_{filename.split('_')[-1]}"
            save_model(vectorizer, vec_filename)
        elif model_type == 'word2vec':
            vec_filename = f"word2vec_model_{filename.split('_')[-1].replace('.pkl', '.bin')}"
            vectorizer.save(os.path.join(MODEL_DIR, vec_filename))
        
        update_metadata(model_type, filename, accuracy)
        
        return JsonResponse({
            'status': 'success',
            'model_type': model_type,
            'accuracy': accuracy,
            'filename': filename
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def load_pretrained_model(request):
    """Load a pretrained model"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        params = json.loads(request.body)
        model_type = params.get('model_type')
        filename = params.get('filename')
        
        if not model_type or not filename:
            return JsonResponse({'error': 'Missing model_type or filename'}, status=400)
        
        # Load model
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            return JsonResponse({'error': 'Model file not found'}, status=404)
            
        model = load_model(filename)
        
        # Load vectorizer
        if model_type in ['tfidf', 'bow', 'lda']:
            vec_filename = f"{model_type}_vectorizer_{filename.split('_')[-1]}"
            vectorizer = load_model(vec_filename)
        elif model_type == 'word2vec':
            vec_filename = f"word2vec_model_{filename.split('_')[-1].replace('.pkl', '.bin')}"
            vectorizer = Word2Vec.load(os.path.join(MODEL_DIR, vec_filename))
        else:
            return JsonResponse({'error': 'Invalid model type'}, status=400)
        
        # Create new active learning session
        session_id = f"pretrained_{int(time.time())}"
        
        X, y = load_imdb_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Transform all training data
        X_vec = transform_text(X_train, vectorizer, model_type)
        
        # Split into labeled and unlabeled pools for active learning
        initial_size = 100  # You can make this configurable from frontend if needed
        indices = np.random.permutation(len(X_train))
        labeled_indices = indices[:initial_size].tolist()
        unlabeled_indices = indices[initial_size:].tolist()
        
        # Store session
        ACTIVE_LEARNING_SESSIONS[session_id] = {
            'model_type': model_type,
            'strategy': 'uncertainty',  # Default strategy
            'batch_size': 5,  # Default batch size
            'vectorizer': vectorizer,
            'X_text': X_train.values,
            'X_vec': X_vec,
            'y': y_train.values,
            'y_test': y_test.values,
            'labeled_indices': labeled_indices,
            'unlabeled_indices': unlabeled_indices,
            'clf': model,
            'history': {
                'accuracy': [accuracy_score(y_test, model.predict(transform_text(X_test, vectorizer, model_type)))],
                'samples': [len(labeled_indices)]
            },
            'pretrained': True  # Flag to indicate this is a pretrained model
        }
        
        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'accuracy': ACTIVE_LEARNING_SESSIONS[session_id]['history']['accuracy'][0],
            'message': 'Pretrained model loaded successfully'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def list_models_api(request):
    """API endpoint to list available models"""
    model_type = request.GET.get('model_type')
    metadata = load_metadata()
    
    if model_type:
        models = metadata.get(model_type, [])
    else:
        models = []
        for mt in metadata:
            models.extend(metadata[mt])
    
    # Sort by accuracy descending and add full filename
    models = sorted(models, key=lambda x: x['accuracy'], reverse=True)
    for model in models:
        model['filename'] = f"{model['filename']}"
    
    return JsonResponse({
        'status': 'success',
        'models': models
    })

@csrf_exempt
def start_active_learning(request):
    """Initialize active learning session"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        params = json.loads(request.body)
        session_id = params.get('session_id', 'default')
        model_type = params.get('model_type', 'tfidf')
        strategy = params.get('strategy', 'uncertainty')
        batch_size = int(params.get('batch_size', 5))
        initial_size = int(params.get('initial_size', 100))
        
        X, y = load_imdb_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize vectorizer
        vectorizer, X_vec = initialize_vectorizer(model_type, X_train)
        
        # Create initial labeled pool
        indices = np.random.permutation(len(X_train))
        labeled_indices = indices[:initial_size]
        unlabeled_indices = indices[initial_size:]
        
        # Initialize classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_vec[labeled_indices], y_train.iloc[labeled_indices])
        
        # Store session
        ACTIVE_LEARNING_SESSIONS[session_id] = {
            'model_type': model_type,
            'strategy': strategy,
            'batch_size': batch_size,
            'vectorizer': vectorizer,
            'X_text': X_train.values,
            'X_vec': X_vec,
            'y': y_train.values,
            'y_test': y_test.values,
            'labeled_indices': labeled_indices.tolist(),
            'unlabeled_indices': unlabeled_indices.tolist(),
            'clf': clf,
            'history': {
                'accuracy': [accuracy_score(y_test, clf.predict(transform_text(X_test, vectorizer, model_type)))],
                'samples': [initial_size]
            },
            'pretrained': False
        }
        
        return JsonResponse({
            'status': 'success',
            'session_id': session_id,
            'initial_accuracy': ACTIVE_LEARNING_SESSIONS[session_id]['history']['accuracy'][0]
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_next_batch(request):
    """Get next batch of samples to label"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        params = json.loads(request.body)
        session_id = params.get('session_id', 'default')
        
        if session_id not in ACTIVE_LEARNING_SESSIONS:
            return JsonResponse({'error': 'Invalid session ID'}, status=400)
            
        session = ACTIVE_LEARNING_SESSIONS[session_id]
        
        # For pretrained models with no unlabeled samples
        if session.get('pretrained', False) and len(session['unlabeled_indices']) == 0:
            return JsonResponse({
                'status': 'success',
                'samples': [],
                'message': 'No unlabeled samples available for pretrained model'
            })
            
        strategy = session['strategy']
        batch_size = session['batch_size']
        unlabeled_indices = np.array(session['unlabeled_indices'])
        X_pool = session['X_vec'][unlabeled_indices]
        clf = session['clf']
        
        # Select samples based on strategy
        if strategy == 'random':
            query_idx = np.random.choice(len(unlabeled_indices), batch_size, replace=False)
        else:
            probs = clf.predict_proba(X_pool)
            if strategy == 'uncertainty':
                scores = 1 - np.max(probs, axis=1)
            elif strategy == 'margin':
                scores = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
            elif strategy == 'entropy':
                scores = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            query_idx = np.argpartition(scores, -batch_size)[-batch_size:]
        
        # Get original indices and text samples
        sample_indices = unlabeled_indices[query_idx].tolist()
        samples = [{
            'id': int(idx),
            'text': session['X_text'][idx]
        } for idx in sample_indices]
        
        # Store current batch
        session['current_batch'] = {
            'query_idx': query_idx.tolist(),
            'sample_indices': sample_indices
        }
        
        return JsonResponse({
            'status': 'success',
            'samples': samples,
            'strategy': strategy
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def submit_labels(request):
    """Submit labeled samples and update model"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    try:
        params = json.loads(request.body)
        session_id = params.get('session_id', 'default')
        labels = params.get('labels', {})
        
        if session_id not in ACTIVE_LEARNING_SESSIONS:
            return JsonResponse({'error': 'Invalid session ID'}, status=400)
            
        session = ACTIVE_LEARNING_SESSIONS[session_id]
        
        if 'current_batch' not in session:
            return JsonResponse({'error': 'No active batch'}, status=400)
            
        # Update labeled data
        labeled_indices = session['labeled_indices']
        unlabeled_indices = session['unlabeled_indices']
        batch_indices = session['current_batch']['sample_indices']
        
        # Add new labels
        for idx in batch_indices:
            labeled_indices.append(idx)
            unlabeled_indices.remove(idx)
        
        # Update y_train with new labels
        for idx, label in labels.items():
            idx = int(idx)
            if idx in batch_indices:
                session['y'][idx] = int(label)
        
        # Retrain model
        clf = clone(session['clf'])
        X_labeled = session['X_vec'][labeled_indices]
        y_labeled = session['y'][labeled_indices]
        clf.fit(X_labeled, y_labeled)
        
        # Evaluate
        X_test_vec = transform_text(session['X_text'][:len(session['y_test'])], session['vectorizer'], session['model_type'])
        acc = accuracy_score(session['y_test'], clf.predict(X_test_vec))
        
        # Update session
        session['clf'] = clf
        session['history']['accuracy'].append(acc)
        session['history']['samples'].append(len(labeled_indices))
        del session['current_batch']
        
        # For pretrained models, maintain all samples as labeled
        if session.get('pretrained', False):
            session['labeled_indices'] = list(range(len(session['X_text'])))
            session['unlabeled_indices'] = []
        
        return JsonResponse({
            'status': 'success',
            'accuracy': acc,
            'samples_labeled': len(labeled_indices),
            'samples_remaining': len(unlabeled_indices),
            'history': session['history']
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def learning_progress(request, session_id='default'):
    """Get current learning progress"""
    if request.method != 'GET':
        return JsonResponse({'error': 'GET method required'}, status=400)
        
    if session_id not in ACTIVE_LEARNING_SESSIONS:
        return JsonResponse({'error': 'Invalid session ID'}, status=400)
        
    session = ACTIVE_LEARNING_SESSIONS[session_id]
    return JsonResponse({
        'status': 'success',
        'history': session['history'],
        'model_type': session['model_type'],
        'strategy': session['strategy'],
        'samples_labeled': len(session['labeled_indices']),
        'samples_remaining': len(session['unlabeled_indices']),
        'pretrained': session.get('pretrained', False)
    })