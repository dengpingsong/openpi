# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the OpenPi policy evaluation script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --policy_port=8000 \
    --lang_instruction="Grab markers and place into pen holder."
```

First start the OpenPi policy server:
```shell
uv run scripts/serve_policy.py --env=DROID
```

Then replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.utils.utils import (
    init_logging,
    log_say,
)

# NOTE:
# Using openpi-client to communicate with remote policy server via WebSocket
from openpi_client import image_tools
from openpi_client import websocket_client_policy

#################################################################################


class OpenPiRobotInferenceClient:
    """Client for communicating with OpenPi remote policy server via WebSocket.

    This currently only supports so100_follower, so101_follower
    modify this code to support other robots with other keys based on your policy configuration.
    """

    def __init__(
        self,
        host="localhost",
        port=8000,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
        resize_size=224,
    ):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        self.resize_size = resize_size
        assert (
            len(robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "

    def get_action(self, observation_dict, lang: str):
        # Construct observation in OpenPi format
        # Try different formats based on the policy expectations
        
        obs_dict = {}

        # Add robot state (unnormalized, will be normalized on server side)
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state"] = state.astype(np.float64)
        
        # Process and resize images for bandwidth efficiency
        processed_images = {}
        for camera_key in self.camera_keys:
            if camera_key in observation_dict:
                img = observation_dict[camera_key]
                # Resize with padding and convert to uint8 to minimize bandwidth
                resized_img = image_tools.resize_with_pad(img, self.resize_size, self.resize_size)
                uint8_img = image_tools.convert_to_uint8(resized_img)
                
                # Keep images in (H, W, C) format for SO101 (server expects this format)
                processed_images[camera_key] = uint8_img
        
        # Format for SO101 policy (all 3 image keys required)
        if "front" in processed_images:
            obs_dict["image"] = processed_images["front"]
        else:
            # Create a black image if front camera not available
            obs_dict["image"] = np.zeros((self.resize_size, self.resize_size, 3), dtype=np.uint8)
            
        if "wrist" in processed_images:
            obs_dict["wrist_image"] = processed_images["wrist"]
        else:
            # Create a black image if wrist camera not available
            obs_dict["wrist_image"] = np.zeros((self.resize_size, self.resize_size, 3), dtype=np.uint8)
            
        # SO101 expects a tactile_image, use black image if not available
        if "tactile" in processed_images:
            obs_dict["tactile_image"] = processed_images["tactile"]
        else:
            obs_dict["tactile_image"] = np.zeros((self.resize_size, self.resize_size, 3), dtype=np.uint8)
        
        # Add language instruction
        obs_dict["prompt"] = lang

        # Show images if requested
        if self.show_images:
            # Images are already in (H, W, C) format for display
            img_dict = {}
            for key in ["image", "wrist_image", "tactile_image"]:
                if key in obs_dict:
                    img_dict[key] = obs_dict[key]
            view_img(img_dict)

        print(f"Sending observation with keys: {list(obs_dict.keys())}")
        print(f"State shape: {obs_dict['state'].shape}")
        for img_key in ["image", "wrist_image", "tactile_image"]:
            if img_key in obs_dict:
                print(f"{img_key} shape: {obs_dict[img_key].shape}")

        # Call the policy server with the current observation
        # This returns an action chunk of shape (action_horizon, action_dim)
        response = self.policy.infer(obs_dict)
        action_chunk = response["actions"]

        # Convert the action chunk to a list of dict[str, float] for lerobot
        lerobot_actions = []
        action_horizon = action_chunk.shape[0]
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: np.ndarray, idx: int
    ) -> dict[str, float]:
        """
        Convert action chunk at given index to a dict[str, float] for robot control.
        
        Args:
            action_chunk: numpy array of shape (action_horizon, action_dim)
            idx: index of the action to extract
            
        Returns:
            Dictionary mapping robot state keys to action values
        """
        action = action_chunk[idx]
        assert len(action) == len(self.robot_state_keys), f"Action dimension {len(action)} doesn't match robot state keys {len(self.robot_state_keys)}"
        
        # Convert the action to dict[str, float]
        action_dict = {key: float(action[i]) for i, key in enumerate(self.robot_state_keys)}
        return action_dict


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the openpi policy server
    policy_port: int = 8000  # port of the openpi policy server (default WebSocket port)
    action_horizon: int = 8  # number of actions to execute from the action chunk
    lang_instruction: str = "Grab pens and place into pen holder."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 60  # timeout in seconds
    show_images: bool = False  # whether to show images
    resize_size: int = 224  # image resize size for policy input


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # Step 2: Initialize the policy
    policy = OpenPiRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
        resize_size=cfg.resize_size,
    )
    log_say(
        "Initializing OpenPi policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    while True:
        # get the realtime image
        observation_dict = robot.get_observation()
        print("observation_dict", observation_dict.keys())
        action_chunk = policy.get_action(observation_dict, language_instruction)

        for i in range(cfg.action_horizon):
            action_dict = action_chunk[i]
            # ss=""
            # for k, v in action_dict.items():
            #     s=f"{np.array(v)}"
            #     ss+=s
            #  print(ss)
            print("action_dict:", {k: action_dict[k] for k in action_dict})
           
            robot.send_action(action_dict)
            time.sleep(0.02)  # Implicitly wait for the action to be executed


if __name__ == "__main__":
    eval()
