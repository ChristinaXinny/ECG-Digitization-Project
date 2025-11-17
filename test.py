#!/usr/bin/env python3
"""Convenience script to run tests from project root."""

import os
import sys
import subprocess

def main():
    """Run test script from tests directory."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, 'tests', 'run_simple_tests.py')

    if not os.path.exists(script_path):
        print(f"Error: Test script not found at {script_path}")
        return 1

    # Set environment variable for Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root

    # Run the test script with all arguments
    cmd = [sys.executable, script_path] + sys.argv[1:]

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)

if __name__ == '__main__':
    sys.exit(main())