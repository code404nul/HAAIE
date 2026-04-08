# Don't use this when use this project with sensitive information. For debug case only !

import json
import os
from datetime import datetime


LOG_FILE = "logs/chat_history.json"


def _load() -> list:
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def _save(data: list):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log(input_text: str, output_text: str):
    """Enregistre une paire input/output dans le fichier JSON."""
    entries = _load()
    entries.append({
        "timestamp": datetime.now().isoformat(),
        "input": input_text,
        "output": output_text,
    })
    _save(entries)