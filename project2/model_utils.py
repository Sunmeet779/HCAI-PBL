# project2/model_utils.py
import os
import json
import pickle
from datetime import datetime
from collections import defaultdict
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

METADATA_FILE = os.path.join(MODEL_DIR, 'metadata.json')

os.makedirs(MODEL_DIR, exist_ok=True)

def load_metadata():
    """Load model metadata from JSON file"""
    if not os.path.exists(METADATA_FILE):
        return defaultdict(list)
    
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return defaultdict(list)

def save_metadata(metadata):
    """Save model metadata to JSON file"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
    except IOError:
        pass

def update_metadata(model_type, filename, accuracy):
    """Update metadata with new model information"""
    metadata = load_metadata()
    metadata.setdefault(model_type, []).append({
        'filename': filename,
        'accuracy': float(accuracy),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_metadata(metadata)

def get_best_model(model_type):
    """Get the best performing model for a given type"""
    metadata = load_metadata()
    if not metadata.get(model_type):
        return None
    return sorted(metadata[model_type], key=lambda x: x['accuracy'], reverse=True)[0]

def get_model_path(filename):
    """Get full path to a model file"""
    return os.path.join(MODEL_DIR, filename)

def save_model(model, filename):
    """Save a model to disk"""
    with open(get_model_path(filename), 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """Load a model from disk"""
    with open(get_model_path(filename), 'rb') as f:
        return pickle.load(f)

def list_models():
    """List all available models with metadata"""
    models = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.pkl') or f.endswith('.bin'):
            parts = f.split('_')
            try:
                model_info = {
                    'filename': f,
                    'type': parts[0],
                    'full_path': get_model_path(f)
                }
                if len(parts) > 1:
                    model_info['accuracy'] = float(parts[1])
                if len(parts) > 2:
                    model_info['date'] = datetime.strptime(parts[2], "%Y%m%d").date()
                if len(parts) > 3:
                    model_info['time'] = parts[3].split('.')[0]
                models.append(model_info)
            except (ValueError, IndexError):
                models.append({
                    'filename': f,
                    'type': 'unknown',
                    'full_path': get_model_path(f)
                })
    return models