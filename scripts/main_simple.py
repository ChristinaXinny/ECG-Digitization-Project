#!/usr/bin/env python3
"""
Simplified main entry point for ECG Digitization Project.

This script provides basic interface for training and inference.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent  # Go up one level from scripts/ to project root
sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ECG Digitization Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference
  python scripts/main_simple.py inference --config configs/inference_config.yaml --input path/to/ecg.png

  # Run training
  python scripts/main_simple.py train --config configs/base.yaml
        """
    )

    parser.add_argument(
        "command",
        choices=["train", "inference"],
        help="Command to run (train/inference)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input file path (for inference)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Output directory"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Model checkpoint path (for inference)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",  # User can override if needed
        help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def run_inference(args, config, logger):
    """Run inference using the inference.py script."""
    logger.info("Starting inference...")

    # Check if input file exists
    if not args.input:
        logger.error("Input file not specified")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Use checkpoint from args or config
    checkpoint_path = args.checkpoint or config.get('INFERENCE', {}).get('CHECKPOINT', 'outputs/stage0_final.pth')

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Please train a model first or specify the correct checkpoint path")
        sys.exit(1)

    # Run the actual inference using load_model.py
    import subprocess

    cmd = [
        sys.executable,
        str(project_root / "scripts" / "load_model.py"),
        "--checkpoint", checkpoint_path,
        "--image", str(input_path),
        "--output", args.output,
        "--device", args.device
    ]

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        logger.info("Inference completed successfully!")
    else:
        logger.error("Inference failed!")
        sys.exit(1)


def run_training(args, config, logger):
    """Run training using the training scripts."""
    logger.info("Starting training...")

    # Use train.py or stage-specific scripts
    train_script = str(project_root / "train.py")

    if Path(train_script).exists():
        import subprocess
        cmd = [sys.executable, train_script]
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            logger.error("Training failed!")
            sys.exit(1)
    else:
        logger.error("Training script not found")
        logger.info("Try using: python scripts/train_stage0.py")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(
        log_dir=config.get('LOG', {}).get('LOG_DIR', 'outputs/logs'),
        log_level=log_level,
        save_to_file=False,  # Simplified logging
        console_output=True
    )

    logger.info(f"Command: {args.command}")

    try:
        if args.command == "inference":
            run_inference(args, config, logger)
        elif args.command == "train":
            run_training(args, config, logger)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in {args.command}: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info(f"{args.command.capitalize()} completed successfully!")


if __name__ == "__main__":
    main()