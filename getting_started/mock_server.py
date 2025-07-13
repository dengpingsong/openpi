#!/usr/bin/env python3
"""
Mock server to test different observation formats.
Helps identify which format the policy expects without needing the actual server.
"""

import asyncio
import json
import numpy as np
import websockets


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return {
                "__numpy__": True,
                "dtype": str(o.dtype),
                "shape": o.shape,
                "data": o.tolist()
            }
        return super().default(o)


async def handle_client(websocket):
    """Handle client connections and observation testing."""
    print("Client connected")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"\n📨 Received observation with keys: {list(data.keys())}")
                
                # Analyze the observation format
                if "state" in data:
                    state = np.array(data["state"])
                    print(f"  State shape: {state.shape}, dtype: {state.dtype}")
                
                # Check image formats
                image_keys = [k for k in data.keys() if "image" in k.lower()]
                print(f"  Image keys: {image_keys}")
                
                for key in image_keys:
                    if key in data and isinstance(data[key], list):
                        img_array = np.array(data[key])
                        print(f"    {key} shape: {img_array.shape}, dtype: {img_array.dtype}")
                
                if "prompt" in data:
                    print(f"  Prompt: {data['prompt']}")
                
                # Mock response - return dummy actions
                response = {
                    "actions": np.random.rand(10, 6).tolist(),  # 10 steps, 6 DOF
                    "status": "success"
                }
                
                print(f"🤖 Sending mock response with actions shape: (10, 6)")
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                error_response = {"error": f"JSON decode error: {e}"}
                await websocket.send(json.dumps(error_response))
            except Exception as e:
                print(f"❌ Processing error: {e}")
                error_response = {"error": f"Processing error: {e}"}
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    """Start the mock server."""
    print("🚀 Starting mock server on localhost:8000")
    print("This server will accept any observation format and return mock actions.")
    print("Use Ctrl+C to stop the server.")
    
    async with websockets.serve(handle_client, "localhost", 8000):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Mock server stopped")
