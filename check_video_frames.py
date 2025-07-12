#!/usr/bin/env python3
"""Check video frame counts to identify problematic episodes."""

import os
import pandas as pd
from pathlib import Path
import subprocess
import json


def check_video_frames(video_path):
    """Check the number of frames in a video file using ffprobe."""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet', 
            '-print_format', 'json', 
            '-show_streams', 
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    return int(stream.get('nb_frames', 0))
        return -1
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return -1


def check_parquet_length(parquet_path):
    """Check the number of rows in a parquet file."""
    try:
        df = pd.read_parquet(parquet_path)
        return len(df)
    except Exception as e:
        print(f"Error reading {parquet_path}: {e}")
        return -1


def main():
    base_path = Path("/hdd/dps/openpi/merge_data/so101_follower")
    data_path = base_path / "data" / "chunk-000"
    video_path = base_path / "videos" / "chunk-000" / "observation.images.front"
    
    problems = []
    
    # Get all episode files
    parquet_files = sorted(data_path.glob("episode_*.parquet"))
    
    print(f"Found {len(parquet_files)} episode files")
    
    for i, parquet_file in enumerate(parquet_files):
        episode_name = parquet_file.stem  # e.g., "episode_000000"
        video_file = video_path / f"{episode_name}.mp4"
        
        print(f"Checking {i+1}/{len(parquet_files)}: {episode_name}")
        
        if not video_file.exists():
            print(f"Missing video file: {video_file}")
            problems.append(episode_name)
            continue
            
        # Check parquet length
        parquet_length = check_parquet_length(parquet_file)
        
        # Check video frame count
        video_frames = check_video_frames(video_file)
        
        print(f"  parquet={parquet_length}, video={video_frames}")
        
        # Check for potential issues
        if parquet_length != video_frames:
            print(f"  WARNING: Length mismatch!")
            problems.append(episode_name)
        
        if video_frames == -1 or parquet_length == -1:
            print(f"  ERROR: Could not read file!")
            problems.append(episode_name)
            
        # Check for the specific problematic frame count
        if video_frames == 283:
            print(f"  FOUND PROBLEM: This episode has exactly 283 frames!")
            problems.append(f"{episode_name}_283_frames")
    
    if problems:
        print(f"\nProblematic episodes: {problems}")
        
        # Save problematic episodes to a file for easy removal
        with open("problematic_episodes.txt", "w") as f:
            for problem in problems:
                f.write(f"{problem}\n")
        print("Problematic episodes saved to problematic_episodes.txt")
    else:
        print("\nNo obvious problems found!")


if __name__ == "__main__":
    main()
