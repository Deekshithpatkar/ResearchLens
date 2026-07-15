import json
from typing import Dict, Optional
from backend.config import METADATA_FILE, get_user_data_dir

def load_metadata(user_id: Optional[str] = None) -> Dict:
    """Load or create metadata file for a user or legacy global database."""
    if user_id:
        meta_file = get_user_data_dir(user_id) / "metadata.json"
    else:
        meta_file = METADATA_FILE

    if meta_file.exists():
        with open(meta_file, 'r', encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"papers": {}}
    return {"papers": {}}

def save_metadata(metadata: Dict, user_id: Optional[str] = None):
    """Save metadata to user-scoped file or legacy global database."""
    if user_id:
        meta_file = get_user_data_dir(user_id) / "metadata.json"
    else:
        meta_file = METADATA_FILE
    with open(meta_file, 'w', encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
