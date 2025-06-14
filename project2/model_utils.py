import os
import json
from datetime import datetime

MODELS_DIR = "project2/models"
METADATA_FILE = os.path.join(MODELS_DIR, "metadata.json")

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def update_metadata(rep_type, filename, accuracy):
    metadata = load_metadata()
    entry = {
        "filename": filename,
        "accuracy": accuracy,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if rep_type not in metadata:
        metadata[rep_type] = []
    metadata[rep_type].append(entry)
    save_metadata(metadata)

def get_best_model(rep_type):
    metadata = load_metadata()
    if rep_type not in metadata or len(metadata[rep_type]) == 0:
        return None
    # Sort by accuracy descending
    sorted_models = sorted(metadata[rep_type], key=lambda x: x["accuracy"], reverse=True)
    return sorted_models[0]  # best accuracy model info
