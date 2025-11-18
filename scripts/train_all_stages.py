"""Train all three stages of the ECG digitization pipeline sequentially."""

import os
import sys
import time
import subprocess
import argparse

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_with_time(message):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def run_stage(stage_name, script_path, extra_args=None):
    """Run a training stage."""
    print_with_time(f"Starting {stage_name}...")
    print("=" * 60)

    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    try:
        # Run the script
        result = subprocess.run(cmd, cwd=os.path.dirname(script_path), capture_output=False)

        if result.returncode == 0:
            print_with_time(f"{stage_name} completed successfully!")
        else:
            print_with_time(f"{stage_name} failed with return code: {result.returncode}")
            return False

    except Exception as e:
        print_with_time(f"Error running {stage_name}: {e}")
        return False

    print("=" * 60)
    return True

def main():
    """Main function to train all stages."""
    parser = argparse.ArgumentParser(description="Train all ECG digitization stages")
    parser.add_argument("--stages", nargs='+', choices=['0', '1', '2', 'all'],
                       default=['all'], help="Stages to train (0, 1, 2, or all)")
    parser.add_argument("--stop-on-error", action="store_true",
                       help="Stop training if any stage fails")

    args = parser.parse_args()

    print("ECG Digitization - All Stages Training Pipeline")
    print("=" * 60)

    # Define stage configurations
    stages = {
        '0': {
            'name': 'Stage 0 - Image Standardization and Keypoint Detection',
            'script': 'train_stage0.py'
        },
        '1': {
            'name': 'Stage 1 - Image Rectification and Grid Detection',
            'script': 'train_stage1.py'
        },
        '2': {
            'name': 'Stage 2 - Signal Digitization and Time Series Extraction',
            'script': 'train_stage2.py'
        }
    }

    # Determine which stages to run
    if 'all' in args.stages:
        stages_to_run = ['0', '1', '2']
    else:
        stages_to_run = args.stages

    print_with_time(f"Training stages: {', '.join(stages_to_run)}")
    print()

    # Check if scripts exist
    scripts_dir = os.path.dirname(__file__)
    for stage_id in stages_to_run:
        script_path = os.path.join(scripts_dir, stages[stage_id]['script'])
        if not os.path.exists(script_path):
            print_with_time(f"ERROR: Script not found: {script_path}")
            return

    start_time = time.time()
    successful_stages = []
    failed_stages = []

    # Run each stage
    for stage_id in stages_to_run:
        stage_info = stages[stage_id]
        script_path = os.path.join(scripts_dir, stage_info['script'])

        success = run_stage(stage_info['name'], script_path)

        if success:
            successful_stages.append(stage_id)
        else:
            failed_stages.append(stage_id)
            if args.stop_on_error:
                print_with_time("Stopping training due to error in previous stage")
                break

    # Summary
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print_with_time(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Successful stages: {', '.join(successful_stages) if successful_stages else 'None'}")
    print(f"Failed stages: {', '.join(failed_stages) if failed_stages else 'None'}")

    if successful_stages:
        print("\nGenerated checkpoints:")
        for stage_id in successful_stages:
            checkpoint_dir = f"./outputs/stage{stage_id}_checkpoints"
            if os.path.exists(checkpoint_dir):
                print(f"  Stage {stage_id}: {checkpoint_dir}")
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                for checkpoint in sorted(checkpoints):
                    print(f"    - {checkpoint}")

    print(f"\nTraining pipeline completed in {total_time/60:.2f} minutes")

if __name__ == "__main__":
    main()