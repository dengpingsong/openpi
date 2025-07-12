#!/usr/bin/env python3
"""
修复LeRobot数据集中错误的episode_index字段。
确保每个parquet文件中的episode_index字段与文件名匹配。
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def fix_episode_indices(data_dir: Path):
    """修复数据目录中所有parquet文件的episode_index字段"""
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    parquet_files = sorted(data_dir.glob('episode_*.parquet'))
    print(f"Found {len(parquet_files)} parquet files")
    
    fixed_count = 0
    
    for ep_file in parquet_files:
        # 从文件名提取expected episode index
        ep_num = int(ep_file.stem.split('_')[1])
        
        try:
            # 读取parquet文件
            df = pd.read_parquet(ep_file)
            
            # 检查当前的episode_index值
            current_episode_indices = df['episode_index'].unique()
            expected_idx = ep_num
            
            if len(current_episode_indices) != 1 or current_episode_indices[0] != expected_idx:
                print(f"Fixing {ep_file.name}: {list(current_episode_indices)} -> {expected_idx}")
                
                # 修复episode_index字段
                df['episode_index'] = expected_idx
                
                # 保存修复后的文件
                df.to_parquet(ep_file, index=False)
                fixed_count += 1
            else:
                if ep_num < 5 or ep_num % 20 == 0:  # 显示前几个和每20个的状态
                    print(f"{ep_file.name}: OK (episode_index = {expected_idx})")
                    
        except Exception as e:
            print(f"Error processing {ep_file.name}: {e}")
    
    print(f"\nFixed {fixed_count} files")
    return fixed_count

if __name__ == "__main__":
    data_dir = Path("/hdd/dps/openpi/merge_data/so101_follower/data/chunk-000")
    fix_episode_indices(data_dir)
