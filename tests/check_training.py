#!/usr/bin/env python3
"""Training status and checkpoint checker for ECG digitization models."""

import os
import torch
import glob
from datetime import datetime

def check_checkpoints(checkpoint_dir: str = "./outputs/stage0_checkpoints/"):
    """
    Check training progress and model checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint files
    """
    print("=== ECG Training Status Checker ===")
    print(f"Checking directory: {checkpoint_dir}")
    print()

    if not os.path.exists(checkpoint_dir):
        print(f" Checkpoint directory not found: {checkpoint_dir}")
        return

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))

    if not checkpoint_files:
        print(" No checkpoint files found")
        return

    print(f" Found {len(checkpoint_files)} checkpoint files:")

    best_checkpoint = None
    best_loss = float('inf')
    latest_checkpoint = None
    latest_epoch = 0

    for checkpoint_file in sorted(checkpoint_files):
        filename = os.path.basename(checkpoint_file)
        file_size = os.path.getsize(checkpoint_file) / (1024 * 1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_file))

        try:
            # Load checkpoint info
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            epoch = checkpoint.get('epoch', 'Unknown')
            loss = checkpoint.get('loss', 'Unknown')

            print(f"   {filename}")
            print(f"     Epoch: {epoch}")
            print(f"     Loss: {loss}")
            print(f"     Size: {file_size:.1f} MB")
            print(f"     Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Track best and latest
            if isinstance(loss, (int, float)) and loss < best_loss:
                best_loss = loss
                best_checkpoint = checkpoint_file

            if isinstance(epoch, int) and epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = checkpoint_file

            print()

        except Exception as e:
            print(f"   {filename} - Error loading: {e}")
            print()

    # Summary
    print("=== Summary ===")
    if best_checkpoint:
        print(f" Best model: {os.path.basename(best_checkpoint)} (Loss: {best_loss:.6f})")

    if latest_checkpoint:
        print(f" Latest checkpoint: {os.path.basename(latest_checkpoint)} (Epoch: {latest_epoch})")

    # Check if training is complete
    if "stage0_final.pth" in checkpoint_files:
        print(" Training completed (final model found)")
    elif "stage0_best.pth" in checkpoint_files:
        print(" Training in progress or completed (best model found)")
    else:
        print(" Training likely in progress")

def list_available_models():
    """List all available model checkpoints."""
    print("\n=== Available Models ===")

    models_dir = "./outputs/"
    if not os.path.exists(models_dir):
        print("No outputs directory found")
        return

    # Find all stage directories
    stage_dirs = [d for d in os.listdir(models_dir)
                  if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("stage")]

    if not stage_dirs:
        print("No stage checkpoint directories found")
        return

    for stage_dir in sorted(stage_dirs):
        stage_path = os.path.join(models_dir, stage_dir)
        checkpoint_files = glob.glob(os.path.join(stage_path, "*.pth"))

        if checkpoint_files:
            print(f"\n {stage_dir}/")
            for checkpoint_file in sorted(checkpoint_files):
                filename = os.path.basename(checkpoint_file)
                print(f"    {filename}")

def test_checkpoint(checkpoint_path: str):
    """
    Test if a checkpoint can be loaded successfully.

    Args:
        checkpoint_path: Path to checkpoint file
    """
    print(f"\n=== Testing Checkpoint: {checkpoint_path} ===")

    if not os.path.exists(checkpoint_path):
        print(f" File not found: {checkpoint_path}")
        return

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        print(" Checkpoint loaded successfully")
        print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   Loss: {checkpoint.get('loss', 'Unknown')}")

        # Check for required components
        required_keys = ['model_state_dict', 'config']
        missing_keys = [key for key in required_keys if key not in checkpoint]

        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        else:
            print(" All required keys present")

        # Try to create model
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from models import Stage0Net

            config = checkpoint['config']
            model = Stage0Net(config)
            model.load_state_dict(checkpoint['model_state_dict'])

            print(" Model architecture verified")
            print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        except Exception as e:
            print(f" Model loading failed: {e}")

    except Exception as e:
        print(f" Checkpoint loading failed: {e}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="ECG Training Status Checker")
    parser.add_argument("--dir", type=str, default="./outputs/stage0_checkpoints/",
                       help="Checkpoint directory to check")
    parser.add_argument("--test", type=str,
                       help="Test specific checkpoint file")
    parser.add_argument("--list", action="store_true",
                       help="List all available models")

    args = parser.parse_args()

    if args.test:
        test_checkpoint(args.test)
    else:
        check_checkpoints(args.dir)

        if args.list:
            list_available_models()


if __name__ == "__main__":
    main()