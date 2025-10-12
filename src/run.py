"""
Simple entry point for DR detection system.
"""

import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run training
from src.train import main

if __name__ == "__main__":
    print("Starting DR Detection System...")
    print("Training model...")
    main()