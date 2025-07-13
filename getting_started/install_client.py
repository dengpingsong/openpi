#!/usr/bin/env python3
"""
Install script for openpi-client package.
Run this script to install the openpi-client package in your robot environment.
"""

import subprocess
import sys
import os

def install_openpi_client():
    """Install the openpi-client package."""
    # Get the path to the openpi-client package
    openpi_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client_path = os.path.join(openpi_root, "packages", "openpi-client")
    
    if not os.path.exists(client_path):
        print(f"Error: openpi-client package not found at {client_path}")
        return False
    
    print(f"Installing openpi-client from {client_path}")
    
    try:
        # Install in editable mode
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", client_path], check=True)
        print("✅ openpi-client installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing openpi-client: {e}")
        return False

if __name__ == "__main__":
    install_openpi_client()
