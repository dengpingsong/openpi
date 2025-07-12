"""Compute normalization statistics for a config, with safety checks.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory. This version includes safety checks to skip problematic episodes.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示 FATAL 错误，屏蔽 INFO、WARNING、ERROR


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class SafeDataset:
    """A wrapper that safely iterates through a dataset, skipping problematic samples."""
    
    def __init__(self, dataset, problematic_episodes=None):
        self.dataset = dataset
        self.problematic_episodes = set(problematic_episodes or [])
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
        
    def safe_iter(self, max_attempts=None):
        """Safely iterate through the dataset, skipping problematic samples."""
        max_attempts = max_attempts or len(self.dataset)
        successful = 0
        skipped = 0
        
        for i in range(min(max_attempts, len(self.dataset))):
            try:
                item = self.dataset[i]
                successful += 1
                yield item
            except (RuntimeError, IndexError, KeyError) as e:
                skipped += 1
                print(f"Skipping sample {i} due to error: {e}")
                continue
                
        print(f"Successfully processed {successful} samples, skipped {skipped} problematic samples")


def create_torch_dataloader_safe(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
) -> tuple[SafeDataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    # Create dataset with full action_horizon for proper training compatibility
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    
    # Read problematic episodes list
    problematic_episodes = []
    try:
        with open("problematic_episodes.txt", "r") as f:
            problematic_episodes = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("No problematic_episodes.txt found, proceeding without filtering")
    
    safe_dataset = SafeDataset(dataset, problematic_episodes)
    
    if max_frames is not None and max_frames < len(dataset):
        num_samples = max_frames
    else:
        num_samples = len(dataset)
    
    return safe_dataset, num_samples


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    
    safe_dataset, num_samples = create_torch_dataloader_safe(
        data_config, config.model.action_horizon, config.batch_size, config.model, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    processed_samples = 0
    batch_data = {key: [] for key in keys}
    
    # Manually create batches from safe iteration
    for item in tqdm.tqdm(safe_dataset.safe_iter(num_samples), desc="Computing stats"):
        # Collect data for batch
        for key in keys:
            batch_data[key].append(np.asarray(item[key]))
        
        # Process when we have a full batch or at the end
        if len(batch_data[keys[0]]) >= config.batch_size:
            # Process the batch
            for key in keys:
                batch_array = np.stack(batch_data[key])
                # Take first element of each batch item (removing batch dimension)
                values = batch_array[:, 0] if batch_array.ndim > 2 else batch_array
                stats[key].update(values.reshape(-1, values.shape[-1]))
            
            processed_samples += len(batch_data[keys[0]])
            # Reset batch data
            batch_data = {key: [] for key in keys}
    
    # Process remaining data in incomplete batch
    if batch_data[keys[0]]:
        for key in keys:
            batch_array = np.stack(batch_data[key])
            values = batch_array[:, 0] if batch_array.ndim > 2 else batch_array
            stats[key].update(values.reshape(-1, values.shape[-1]))
        processed_samples += len(batch_data[keys[0]])

    print(f"Successfully processed {processed_samples} samples for norm stats computation")
    
    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    if data_config.repo_id is not None:
        output_path = config.assets_dirs / data_config.repo_id
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
    else:
        print("No repo_id specified, skipping save")


if __name__ == "__main__":
    tyro.cli(main)
