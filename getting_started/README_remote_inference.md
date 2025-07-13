# OpenPi Robot Evaluation with Remote Policy Server

This example demonstrates4. **Observation Format**: Updated to match OpenPi's expected format (depends on the policy):
   - **ALOHA/SO101 style**: 
     - Images: `images` dictionary with camera names as keys (C, H, W format)
     - State: `state` (single array)
     - Prompt: `prompt` (language instruction)
   - **DROID style**:
     - Images: `observation/camera_name` (H, W, C format)
     - State: `observation/joint_position`, `observation/gripper_position`
     - Prompt: `prompt` to use OpenPi models remotely with a robot. The code has been updated to use the official `openpi-client` package for WebSocket communication with the remote policy server.

## Setup

### 1. Install the openpi-client package

First, install the openpi-client package in your robot environment:

```bash
cd /path/to/openpi-1/packages/openpi-client
pip install -e .
```

Or use the provided install script:

```bash
python getting_started/install_client.py
```

### 2. Start the remote policy server

Start the OpenPi policy server on your inference machine:

```bash
# For DROID environment
uv run scripts/serve_policy.py --env=DROID

# For ALOHA environment  
uv run scripts/serve_policy.py --env=ALOHA

# For LIBERO environment
uv run scripts/serve_policy.py --env=LIBERO

# Or specify a custom checkpoint
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

The server will start on port 8000 by default.

## Usage

### Test robot connectivity first

Before running the policy evaluation, test that your robot is working:

```bash
python -m lerobot.replay \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --dataset.repo_id=youliangtan/so100-table-cleanup \
    --dataset.episode=2
```

### Test observation format

To verify which observation format your policy server expects, run:

```bash
python getting_started/test_format.py --host=166.111.192.71 --port=8000
```

This will test both ALOHA and DROID formats and tell you which one works.

### Run policy evaluation

Run the evaluation script with your robot configuration:

```bash
python getting_started/examples/eval_lerobot.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=lil_guy \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --policy_host=10.112.209.136 \
    --policy_port=8000 \
    --lang_instruction="Grab markers and place into pen holder." \
    --show_images=true
```

## Key Changes from ZMQ to WebSocket

The code has been updated with the following key changes:

1. **Client Library**: Now uses `openpi_client` instead of the custom ZMQ service
2. **Port**: Default port changed from 5555 to 8000 (WebSocket server default)
3. **Observation Format**: Updated to match OpenPi's expected format:
   - Images: `observation/camera_name_image` (e.g., `observation/wrist_image`)
   - State: `observation/state` (single array)
   - Prompt: `prompt` (language instruction)
4. **Image Processing**: Images are automatically resized and converted to uint8 for bandwidth efficiency
5. **Response Format**: Actions are returned as a numpy array of shape `(action_horizon, action_dim)`

## Configuration Options

- `--policy_host`: IP address of the policy server (default: localhost)
- `--policy_port`: Port of the policy server (default: 8000)
- `--action_horizon`: Number of actions to execute from each action chunk (default: 8)
- `--resize_size`: Image resize size for policy input (default: 224)
- `--show_images`: Whether to display camera images during execution
- `--lang_instruction`: Task description for the robot

## Troubleshooting

1. **Import errors**: Make sure openpi-client is installed: `pip install -e packages/openpi-client`
2. **Connection refused**: Ensure the policy server is running and accessible
3. **Image format errors**: Verify camera indices and image dimensions
4. **Action dimension mismatch**: Check that robot_state_keys match your robot configuration
5. **KeyError: 'state'**: This usually means the policy expects ALOHA-style format but received DROID-style format
6. **KeyError: 'image'/'wrist_image'/'tactile_image'**: SO101 policy requires all 3 image keys. Use black images for missing sensors.
7. **PIL Image.fromarray TypeError**: Images must be in (H, W, C) format, not (C, H, W) format for SO101 policies
8. **Wrong image format**: SO101 expects (H, W, C) format with uint8 values [0-255]

### Policy Format Requirements

Different OpenPi policies expect different input formats:

**ALOHA Style** (e.g., pi0_aloha_sim, pi0_aloha_*):
```python
{
    "state": np.array([...]),  # Robot joint states
    "images": {
        "cam_name": np.array([C, H, W]),  # Images in channel-first format
        ...
    },
    "prompt": "task description"
}
```

**SO101 Style** (e.g., pi0_fast_so101_local):
```python
{
    "state": np.array([...]),  # Robot joint states (5 dimensions for SO101)
    "image": np.array([H, W, C]),  # Front camera in channel-last format
    "wrist_image": np.array([H, W, C]),  # Wrist/USB camera in channel-last format  
    "tactile_image": np.array([H, W, C]),  # Tactile sensor (gel28w2) in channel-last format
    "prompt": "task description"
}
```

**Note**: All three image keys (`image`, `wrist_image`, `tactile_image`) are required for SO101. If you don't have all sensors, fill missing ones with black images of the same dimensions.

**DROID Style** (e.g., pi0_droid, pi0_fast_droid):
```python
{
    "observation/exterior_image_1_left": np.array([H, W, C]),  # Images in channel-last format
    "observation/wrist_image_left": np.array([H, W, C]),
    "observation/joint_position": np.array([...]),
    "observation/gripper_position": np.array([...]),
    "prompt": "task description"
}
```

The current implementation automatically detects and uses the appropriate format based on the policy type.

## Supported Robots

Currently tested with:
- so100_follower
- so101_follower

To support other robots, modify the `robot_state_keys` and camera configuration in the evaluation script.
