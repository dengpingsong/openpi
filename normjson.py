import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 修改为你的实际路径
DATA_DIR = Path("./merge_data/data/chunk-000")
SAVE_DIR = Path("./merge_data_assets/so101_follower")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = ["observation.state", "action"]

all_stats = {}

for field in FIELDS:
    values = []

    print(f"Processing field: {field}")
    for file in tqdm(sorted(DATA_DIR.glob("*.parquet"))):
        df = pd.read_parquet(file)
        if field not in df.columns:
            print(f"Warning: {field} not in {file}")
            continue

        field_data = np.stack(df[field].dropna().to_numpy())
        values.append(field_data)

    if not values:
        print(f"No data found for {field}, skipping.")
        continue

    values = np.concatenate(values, axis=0)
    mean = values.mean(axis=0).tolist()
    std = values.std(axis=0).tolist()

    all_stats[field] = {
        "mean": mean,
        "std": std,
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }

# 保存为 LeRobot 使用的 norm_stats.json 格式
with open(SAVE_DIR / "norm_stats.json", "w") as f:
    json.dump(all_stats, f, indent=2)

print(f"\n✅ norm_stats.json saved to: {SAVE_DIR/'norm_stats.json'}")