#!/usr/bin/env python3
"""
Test script to verify the observation format for OpenPi remote inference.
"""

import numpy as np
from openpi_client import websocket_client_policy


def test_aloha_format(host="localhost", port=8000):
    """Test ALOHA-style observation format."""
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        
        # Create test observation using the exact format from aloha_policy.py make_aloha_example()
        obs = {
            "state": np.ones((14,)),
            "images": {
                "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            },
            "prompt": "do something",
        }
        
        print("Testing ALOHA format (based on aloha_policy.py)...")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"State shape: {obs['state'].shape}")
        print(f"Images keys: {list(obs['images'].keys())}")
        for cam_name, img in obs["images"].items():
            print(f"  {cam_name} shape: {img.shape} dtype: {img.dtype}")
        
        response = client.infer(obs)
        print("✅ ALOHA format successful!")
        print(f"Response keys: {list(response.keys())}")
        if "actions" in response:
            print(f"Actions shape: {response['actions'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ ALOHA format failed: {e}")
        return False


def test_so101_format(host="localhost", port=8000):
    """Test SO101-style observation format."""
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        
        # Create test observation using the SO101 example format
        obs = make_so101_example()
        
        print("Testing SO101 format (based on so101_policy.py)...")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"State shape: {obs['state'].shape}")
        print(f"Image shape: {obs['image'].shape} dtype: {obs['image'].dtype}")
        print(f"Wrist image shape: {obs['wrist_image'].shape} dtype: {obs['wrist_image'].dtype}")
        print(f"Tactile image shape: {obs['tactile_image'].shape} dtype: {obs['tactile_image'].dtype}")
        
        response = client.infer(obs)
        print("✅ SO101 format successful!")
        print(f"Response keys: {list(response.keys())}")
        if "actions" in response:
            print(f"Actions shape: {response['actions'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ SO101 format failed: {e}")
        return False


def test_so101_format_minimal(host="localhost", port=8000):
    """Test SO101-style observation format with minimal required fields (fallback)."""
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        
        # Create test observation using minimal SO101 format
        obs = make_so101_minimal_example()
        
        print("Testing SO101 format (minimal - front camera only)...")
        print(f"Observation keys: {list(obs.keys())}")
        print(f"State shape: {obs['state'].shape}")
        print(f"Image shape: {obs['image'].shape} dtype: {obs['image'].dtype}")
        
        response = client.infer(obs)
        print("✅ SO101 minimal format successful!")
        print(f"Response keys: {list(response.keys())}")
        if "actions" in response:
            print(f"Actions shape: {response['actions'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ SO101 minimal format failed: {e}")
        return False


def test_droid_format(host="localhost", port=8000):
    """Test DROID-style observation format based on droid_policy.py make_droid_example()."""
    try:
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        
        # Create test observation using the exact format from droid_policy.py make_droid_example()
        obs = {
            "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.rand(7),
            "observation/gripper_position": np.random.rand(1),
            "prompt": "do something",
        }
        
        print("Testing DROID format (based on droid_policy.py)...")
        print(f"Observation keys: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key} shape: {value.shape} dtype: {value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        response = client.infer(obs)
        print("✅ DROID format successful!")
        print(f"Response keys: {list(response.keys())}")
        if "actions" in response:
            print(f"Actions shape: {response['actions'].shape}")
        return True
        
    except Exception as e:
        print(f"❌ DROID format failed: {e}")
        return False


def make_so101_example() -> dict:
    """Creates a random input example for the SO101 policy, based on so101_policy.py."""
    # Create properly formatted images - ensure they're 3D arrays (H, W, C)
    def create_image():
        img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
        return img
    
    return {
        "image": create_image(),  # Front camera (H, W, C)
        "wrist_image": create_image(),  # USB camera (H, W, C)  
        "tactile_image": create_image(),  # Gel28w2 tactile sensor (H, W, C)
        "state": np.random.rand(5).astype(np.float64),  # 5D robot state
        "prompt": "do something",
    }


def make_so101_minimal_example() -> dict:
    """Creates a minimal SO101 example with required keys, filling missing images with zeros."""
    # Create properly formatted images
    def create_image():
        return np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    
    def create_black_image():
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    return {
        "image": create_image(),  # Front camera (real)
        "wrist_image": create_black_image(),  # Dummy wrist camera 
        "tactile_image": create_black_image(),  # Dummy tactile sensor
        "state": np.random.rand(5).astype(np.float64),  # 5D robot state
        "prompt": "do something",
    }


def test_so101_format_variations(host="localhost", port=8000):
    """Test different variations of SO101 format to find what the server expects."""
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    
    variations = [
        # Variation 1: Only image + state + prompt
        {
            "name": "SO101 v1 (image only)",
            "data": {
                "image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "state": np.random.rand(5).astype(np.float64),
                "prompt": "do something",
            }
        },
        # Variation 2: Add wrist_image
        {
            "name": "SO101 v2 (image + wrist)",
            "data": {
                "image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "wrist_image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "state": np.random.rand(5).astype(np.float64),
                "prompt": "do something",
            }
        },
        # Variation 3: All three images
        {
            "name": "SO101 v3 (all images)",
            "data": {
                "image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "wrist_image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "tactile_image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "state": np.random.rand(5).astype(np.float64),
                "prompt": "do something",
            }
        },
        # Variation 4: Try 6D state instead of 5D
        {
            "name": "SO101 v4 (6D state)",
            "data": {
                "image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
                "state": np.random.rand(6).astype(np.float64),
                "prompt": "do something",
            }
        },
    ]
    
    print("Testing SO101 format variations...")
    print("=" * 50)
    
    for variation in variations:
        try:
            print(f"\nTesting {variation['name']}:")
            obs = variation['data']
            print(f"  Keys: {list(obs.keys())}")
            print(f"  State shape: {obs['state'].shape}")
            
            response = client.infer(obs)
            print(f"  ✅ SUCCESS! Actions shape: {response['actions'].shape}")
            return variation['name'], True
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    return None, False


def main():
    """Test different formats and report which one works."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenPi observation formats")
    parser.add_argument("--host", default="localhost", help="Policy server host")
    parser.add_argument("--port", type=int, default=8000, help="Policy server port")
    parser.add_argument("--test-variations", action="store_true", help="Test SO101 format variations")
    
    args = parser.parse_args()
    
    print(f"Testing connection to {args.host}:{args.port}")
    print("=" * 50)
    
    if args.test_variations:
        # Test SO101 variations to find the working format
        working_format, success = test_so101_format_variations(args.host, args.port)
        if success:
            print(f"\n🎯 Found working SO101 format: {working_format}")
        else:
            print("\n❌ No SO101 format variation worked.")
        return
    
    aloha_success = test_aloha_format(args.host, args.port)
    print()
    so101_success = test_so101_format(args.host, args.port)
    print()
    so101_minimal_success = test_so101_format_minimal(args.host, args.port)
    print()
    droid_success = test_droid_format(args.host, args.port)
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"ALOHA format: {'✅ Works' if aloha_success else '❌ Failed'}")
    print(f"SO101 format (full): {'✅ Works' if so101_success else '❌ Failed'}")
    print(f"SO101 format (minimal): {'✅ Works' if so101_minimal_success else '❌ Failed'}")
    print(f"DROID format: {'✅ Works' if droid_success else '❌ Failed'}")
    
    if so101_success:
        print("\n🎯 Use SO101 format (full with 3 images) in your robot code!")
    elif so101_minimal_success:
        print("\n🎯 Use SO101 format (minimal with front camera only) in your robot code!")
    elif aloha_success:
        print("\n🎯 Use ALOHA format (multiple images) in your robot code!")
    elif droid_success:
        print("\n🎯 Use DROID format in your robot code!")
    else:
        print("\n❌ None of the formats worked. Check your server configuration.")
        print("💡 Try running with --test-variations to debug SO101 format issues.")


if __name__ == "__main__":
    main()
