import pandas as pd
from pathlib import Path

data_dir = Path('/hdd/dps/openpi/merge_data/so101_follower/data/chunk-000')
problematic_episodes = []

for ep_file in sorted(data_dir.glob('episode_*.parquet')):
    ep_num = int(ep_file.stem.split('_')[1])
    df = pd.read_parquet(ep_file)
    
    # 检查episode_index字段
    episode_indices = df['episode_index'].unique()
    expected_idx = ep_num
    
    if len(episode_indices) != 1 or episode_indices[0] != expected_idx:
        print(f'{ep_file.name}: Expected {expected_idx}, got {episode_indices}')
        problematic_episodes.append(ep_num)
    elif ep_num < 5:  # 显示前几个正常的作为参考
        print(f'{ep_file.name}: OK (episode_index = {episode_indices[0]})')

print(f'\\nTotal problematic episodes: {len(problematic_episodes)}')
if problematic_episodes:
    print(f'Problematic episode numbers: {problematic_episodes[:10]}...' if len(problematic_episodes) > 10 else f'Problematic episode numbers: {problematic_episodes}')
