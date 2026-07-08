import json
from typing import Dict
from backend.config import METADATA_FILE

def load_metadata() -> Dict:
    """Load or create metadata file"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"papers": {}}
    return {"papers": {}}

def save_metadata(metadata: Dict):
    """Save metadata to file"""
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
