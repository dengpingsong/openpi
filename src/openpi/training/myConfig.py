import dataclasses
import pathlib
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfigFactory, DataConfig, ModelTransformFactory
from etils import epath

@dataclasses.dataclass(frozen=True)
class LeRobotSo101DataConfig(DataConfigFactory):
    """
    DataConfig for the so101_follower robot with multiple image modalities (front, usb, gel28w2).
    """

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repacks your dataset keys to standard keys expected by inference pipeline
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform({
                    "observation.images.front": "image",              # main camera
                    "observation.images.usb": "wrist_image",         # optional: wrist cam
                    "observation.images.gel28w2": "tactile_image",   # optional: tactile cam
                    "observation.state": "state",                    # joint states
                    "action": "actions",                             # joint actions
                    "prompt": "prompt",                              # optional
                })
            ]
        )

        # Main input/output transforms
        data_transforms = _transforms.Group(
            inputs=[_transforms.DefaultInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type)],
            outputs=[_transforms.DefaultOutputs()],
        )

        # Apply delta transform to first 5 joints (gripper remains absolute)
        delta_action_mask = _transforms.make_bool_mask(5, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Tokenizer and model-level transforms
        model_transforms = ModelTransformFactory()(model_config)

        # Load norm_stats from: ./merge_data_assets/so101_follower/norm_stats.json
        return dataclasses.replace(
            self.create_base_config(assets_dirs),  # will use assets_dir + asset_id to find norm_stats
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
        
        
