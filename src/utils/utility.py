from pathlib import Path

def path_creator(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)