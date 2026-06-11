import json
from pathlib import Path

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Build full path to data file
DATA_PATH = BASE_DIR / "data" / "shl_catalog.json"


def load_catalog():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_catalog()

    print(f"Loaded {len(data)} assessments")

    print("\nFirst assessment:")

    print(data[0]["name"])