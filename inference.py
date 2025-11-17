#!/usr/bin/env python3
"""Convenience script to run inference from project root."""

import os
import sys
import subprocess

def main():
    """Run inference script from scripts directory."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, 'scripts', 'load_model.py')

    if not os.path.exists(script_path):
        print(f"Error: Inference script not found at {script_path}")
        return 1

    # Run the inference script with all arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

if __name__ == '__main__':
    sys.exit(main())