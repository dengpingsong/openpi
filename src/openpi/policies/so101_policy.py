"""Policy transforms for SO101 follower robot."""

import dataclasses

import numpy as np

import openpi.models.model as _model
import openpi.transforms as transforms


def _parse_image(image):
    """Parse image array to uint8 (H,W,C) format."""
    # Convert tensor to numpy if needed
    if hasattr(image, 'numpy'):  # PyTorch tensor
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        image = np.asarray(image)
    
    if image.dtype == np.uint8 and len(image.shape) == 3:
        return image
    # Convert from float32 (C,H,W) to uint8 (H,W,C)
    if len(image.shape) == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class So101Inputs(transforms.DataTransformFn):
    """Inputs for the SO101 follower policy.
    
    Expected inputs:
    - image: front camera image
    - wrist_image: usb camera image  
    - tactile_image: gel28w2 tactile sensor image
    - state: robot proprioceptive state [5D]
    - actions: action sequence [action_horizon, 6D] (5 joints + 1 gripper)
    """
    
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int
    
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST
        mask_padding = self.model_type == _model.ModelType.PI0
        
        # Pad the proprioceptive state to the model action dimension
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        
        # Parse images to uint8 (H,W,C) format
        front_image = _parse_image(data["image"])
        wrist_image = _parse_image(data["wrist_image"])
        tactile_image = _parse_image(data["tactile_image"])
        
        match self.model_type:
            case _model.ModelType.PI0:
                # Pi0 supports three image inputs: base, left_wrist, right_wrist
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (front_image, wrist_image, tactile_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                # Pi0-FAST uses different naming and doesn't mask padding images
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (front_image, tactile_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        # Add actions if available (during training)
        if "actions" in data:
            inputs["actions"] = np.array(data["actions"])
        
        # Add prompt if available
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]
        
        return inputs


@dataclasses.dataclass(frozen=True)
class So101Outputs(transforms.DataTransformFn):
    """Outputs for the SO101 follower policy."""
    
    def __call__(self, data: dict) -> dict:
        # SO101 has 6 action dimensions (5 joints + 1 gripper)
        return {"actions": np.asarray(data["actions"][:, :6])}
