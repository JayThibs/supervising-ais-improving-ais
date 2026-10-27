from pathlib import Path
from typing import Any, Dict
import json

def serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(i) for i in obj]
    elif hasattr(obj, '__dict__'):
        return serialize_for_json(obj.__dict__)
    return obj

def safe_json_dump(obj: Dict, path: Path) -> None:
    """Safely dump object to JSON file."""
    serialized = serialize_for_json(obj)
    with open(path, 'w') as f:
        json.dump(serialized, f, indent=2)
