"""Main training script for ECG digitization models."""

import os
import sys
import argparse
import torch
from pathlib import Path
from loguru import logger

from configs.base_config import load_config, get_stage_config
from data.dataset import Stage0Dataset, Stage1Dataset, Stage2Dataset
from data.dataset import create_dataloader
from models import Stage0Net, Stage1Net, Stage2Net
from engines.stage_trainer import create_trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ECG digitization models')

    parser.add_argument('--stage', type=str, required=True, choices=['stage0', 'stage1', 'stage2'],
                        help='Which stage to train')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--save-dir', type=str, default='./outputs',
                        help='Directory to save checkpoints and logs')

    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}: {torch.cuda.get_device_name()}")
    else:
        logger.warning("CUDA not available, using CPU")

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(args.save_dir, 'training.log')
    logger.add(log_file, rotation="10 MB", retention="5")


def get_model_and_dataset(stage, config):
    """Get model and dataset for the specified stage."""
    if stage == 'stage0':
        model_class = Stage0Net
        dataset_class = Stage0Dataset
    elif stage == 'stage1':
        model_class = Stage1Net
        dataset_class = Stage1Dataset
    elif stage == 'stage2':
        model_class = Stage2Net
        dataset_class = Stage2Dataset
    else:
        raise ValueError(f"Invalid stage: {stage}")

    return model_class, dataset_class


def create_datasets(dataset_class, config):
    """Create training and validation datasets."""
    # Training dataset
    train_dataset = dataset_class(config, mode="train")

    # Create validation split (20% of training data)
    total_size = len(train_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    # For simplicity, we'll use the same dataset for validation
    # In practice, you might want to create a separate validation split
    val_dataset = dataset_class(config, mode="train")  # Use same for now

    logger.info(f"Training dataset size: {train_size}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Setup environment
    setup_environment(args)

    logger.info(f"Starting training for {args.stage}")
    logger.info(f"Save directory: {args.save_dir}")

    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
        else:
            config = get_stage_config(args.stage)

        # Override config with command line arguments
        if args.epochs:
            config['TRAIN']['EPOCHS'] = args.epochs
        if args.batch_size:
            config['TRAIN']['BATCH_SIZE'] = args.batch_size
        if args.lr:
            config['TRAIN']['LEARNING_RATE'] = args.lr

        # Update device config
        config['DEVICE']['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['DEVICE']['GPU_IDS'] = [args.gpu] if torch.cuda.is_available() else []
        config['DEVICE']['NUM_WORKERS'] = args.num_workers

        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {config['TRAIN']['EPOCHS']}")
        logger.info(f"  Batch size: {config['TRAIN']['BATCH_SIZE']}")
        logger.info(f"  Learning rate: {config['TRAIN']['LEARNING_RATE']}")
        logger.info(f"  Device: {config['DEVICE']['DEVICE']}")

        # Get model and dataset classes
        model_class, dataset_class = get_model_and_dataset(args.stage, config)

        # Create datasets
        train_dataset, val_dataset = create_datasets(dataset_class, config)

        # Create data loaders
        train_loader = create_dataloader(
            train_dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS'],
            pin_memory=True
        )

        val_loader = create_dataloader(
            val_dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=False,
            num_workers=config['DEVICE']['NUM_WORKERS'],
            pin_memory=True
        )

        # Create model
        model = model_class(config)
        logger.info(f"Created {model.__class__.__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Create trainer
        trainer = create_trainer(
            stage=args.stage,
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            resume_from=args.resume
        )

        # Start training
        logger.info("Starting training...")
        trainer.train()

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()