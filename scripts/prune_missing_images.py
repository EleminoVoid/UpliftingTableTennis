import os
import sys
import time
import pandas as pd
from pathlib import Path

# ensure repo root is on sys.path so importing `paths` works
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
import paths

data_root = Path(paths.data_path) / 'tthq'

csv_files = ['table_detection.csv','ball_detection.csv','camera_matrices.csv']

def check_and_prune(csv_name):
    csv_path = data_root / csv_name
    if not csv_path.exists():
        print(f"{csv_path} not found, skipping")
        return
    # these CSVs use semicolon delimiter
    df = pd.read_csv(csv_path, sep=';')
    if 'video' not in df.columns or 'frame' not in df.columns:
        print(f"{csv_name} missing video/frame columns, skipping")
        return
    df['video'] = df['video'].astype(int)
    df['frame'] = df['frame'].astype(int)
    keep_mask = []
    missing = []
    for vid, frm in zip(df['video'], df['frame']):
        img = data_root / f"{vid:02d}" / f"{vid:02d}_{frm:06d}.png"
        if img.exists():
            keep_mask.append(True)
        else:
            keep_mask.append(False)
            missing.append(str(img))
    kept = df[keep_mask]
    backup = csv_path.with_suffix('.csv.bak.' + time.strftime('%Y%m%d-%H%M%S'))
    csv_path.rename(backup)
    kept.to_csv(csv_path, index=False)
    print(f"Pruned {csv_name}: kept {len(kept)}/{len(df)} rows, backup at {backup}")
    if missing:
        print(f"Example missing file: {missing[0]}")

if __name__ == '__main__':
    for c in csv_files:
        check_and_prune(c)
    print('Pruning complete')
