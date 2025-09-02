import pandas as pd
from pathlib import Path

def load_csv(path):
    return pd.read_csv(path)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
