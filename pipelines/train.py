#!/usr/bin/env python3
"""
Training entry script
"""

import sys
import os
import traceback

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distinguisher import Trainer

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python pipelines/train.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        trainer = Trainer(config_path)
        trainer.train()
        print("Training completed!")
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 