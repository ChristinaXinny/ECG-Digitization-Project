#!/usr/bin/env python3
"""
Main entry point for ECG Digitization Project.

This script provides the main interface for training, inference, and evaluation.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent  # Project root is current directory
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config_loader import load_config
from utils.logger import setup_logger
from train import main as train_main
from inference import main as inference_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ECG Digitization Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all stages
  python main.py train --config configs/base.yaml --mode all

  # Train specific stage
  python main.py train --config configs/stage0_config.yaml --mode stage0

  # Run inference
  python main.py inference --config configs/inference_config.yaml --input /path/to/images

  # Evaluate model
  python main.py evaluate --config configs/inference_config.yaml --model outputs/checkpoints/stage0/best.pth
        """
    )

    parser.add_argument(
        "command",
        choices=["train", "inference", "evaluate", "setup"],
        help="Command to execute"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "stage0", "stage1", "stage2", "pipeline"],
        help="Mode for training/inference"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input path for inference"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint for resuming training"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def setup_environment(args):
    """Setup environment based on arguments."""
    # Set GPU
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set random seed
    import torch
    import random
    import numpy as np

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup environment
    setup_environment(args)

    # Setup logging
    log_level = "DEBUG" if args.debug else ("DEBUG" if args.verbose else "INFO")
    logger = setup_logger(
        log_dir=config.get('LOG', {}).get('LOG_DIR', 'outputs/logs'),
        log_level=log_level,
        save_to_file=True,
        console_output=True
    )

    logger.log_system_info()
    logger.info(f"Command: {args.command}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Mode: {args.mode}")

    try:
        if args.command == "train":
            train_main(args, config, logger)

        elif args.command == "inference":
            # Handle inference properly by formatting arguments for load_model.py
            import subprocess

            # Get checkpoint from config or args
            checkpoint = args.model or config.get('PATHS', {}).get('STAGE_WEIGHTS', {}).get('STAGE0', 'outputs/stage0_final.pth')
            input_path = args.input
            output_dir = args.output or 'outputs/'

            if not input_path:
                logger.error("Input path is required for inference")
                sys.exit(1)

            # Build command for load_model.py
            cmd = [
                sys.executable,
                str(project_root / 'scripts' / 'load_model.py'),
                '--checkpoint', checkpoint,
                '--image', input_path,
                '--output', output_dir,
                '--device', 'cuda' if config.get('INFERENCE', {}).get('DEVICE', 'cuda') == 'cuda' else 'cpu'
            ]

            logger.info(f"Running inference with command: {' '.join(cmd)}")
            result = subprocess.run(cmd)

            if result.returncode != 0:
                logger.error("Inference failed!")
                sys.exit(1)
            else:
                logger.info("Inference completed successfully!")

        elif args.command == "evaluate":
            logger.warning("Evaluate functionality not yet implemented")
            # from evaluation import main as evaluate_main
            # evaluate_main(args, config, logger)

        elif args.command == "setup":
            logger.warning("Setup functionality not yet implemented")
            # from scripts.setup_environment import setup_environment_main
            # setup_environment_main(args, config, logger)

        else:
            raise ValueError(f"Unknown command: {args.command}")

    except Exception as e:
        logger.error(f"Error in {args.command}: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    logger.info(f"{args.command.capitalize()} completed successfully!")


if __name__ == "__main__":
    main()