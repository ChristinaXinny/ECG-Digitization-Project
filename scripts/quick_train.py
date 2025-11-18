"""Quick training script for ECG digitization - simple and ready to use."""

import os
import sys
import torch
from torch.utils.data import DataLoader
from loguru import logger

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import Stage0Dataset, Stage1Dataset, Stage2Dataset
from models import Stage0Net, Stage1Net, Stage2Net
from engines.stage_trainer import create_trainer


def quick_train_stage0():
    """Quick training for Stage 0 model."""
    print(" Starting Stage 0 Training...")

    # Configuration
    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 2
        },
        'TRAIN': {
            'BATCH_SIZE': 2,  # Small batch size for testing
            'EPOCHS': 5,      # Few epochs for quick test
            'LEARNING_RATE': 1e-4,
            'LOSS_WEIGHTS': {
                'MARKER_LOSS': 1.0,
                'ORIENTATION_LOSS': 0.5
            }
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet18d',
                'PRETRAINED': False  # Set to False for faster loading
            }
        },
        'DATA': {
            'NORMALIZE': {
                'MEAN': [0.485, 0.456, 0.406],
                'STD': [0.229, 0.224, 0.225]
            }
        },
        'LOG': {
            'LEVEL': 'INFO'
        },
        'CHECKPOINT': {
            'SAVE_DIR': './outputs/stage0_checkpoints'
        }
    }

    try:
        # Create dataset
        print(" Creating dataset...")
        dataset = Stage0Dataset(config, mode="train")
        print(f"[OK] Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS'],
            pin_memory=True
        )

        # Create model
        print(" Creating model...")
        model = Stage0Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created with {total_params:,} parameters")

        # Create trainer
        print(" Setting up trainer...")
        trainer = create_trainer(
            stage='stage0',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None  # No validation for quick test
        )

        # Start training
        print(" Starting training...")
        trainer.train()

        print(" Stage 0 training completed!")
        return True

    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_train_stage1():
    """Quick training for Stage 1 model."""
    print(" Starting Stage 1 Training...")

    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 2
        },
        'TRAIN': {
            'BATCH_SIZE': 2,
            'EPOCHS': 5,
            'LEARNING_RATE': 1e-4,
            'LOSS_WEIGHTS': {
                'MARKER_LOSS': 1.0,
                'GRID_LOSS': 1.0
            }
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet34',
                'PRETRAINED': False
            },
            'DECODER': {
                'HIDDEN_DIMS': [256, 128, 64, 32]
            }
        },
        'DATA': {
            'GRID_CONFIG': {
                'H_LINES': 44,
                'V_LINES': 57
            }
        },
        'CHECKPOINT': {
            'SAVE_DIR': './outputs/stage1_checkpoints'
        }
    }

    try:
        # Create dataset
        print(" Creating dataset...")
        dataset = Stage1Dataset(config, mode="train")
        print(f"[OK] Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS'],
            pin_memory=True
        )

        # Create model
        print(" Creating model...")
        model = Stage1Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created with {total_params:,} parameters")

        # Create trainer
        print(" Setting up trainer...")
        trainer = create_trainer(
            stage='stage1',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None
        )

        # Start training
        print(" Starting training...")
        trainer.train()

        print(" Stage 1 training completed!")
        return True

    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_train_stage2():
    """Quick training for Stage 2 model."""
    print(" Starting Stage 2 Training...")

    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 2
        },
        'TRAIN': {
            'BATCH_SIZE': 1,  # Stage 2 uses larger images, smaller batch
            'EPOCHS': 5,
            'LEARNING_RATE': 1e-4,
            'LOSS_WEIGHTS': {
                'PIXEL_LOSS': 1.0
            }
        },
        'MODEL': {
            'INPUT_SIZE': [1696, 2176],  # Larger input for Stage 2
            'BACKBONE': {
                'NAME': 'resnet34',
                'PRETRAINED': False
            },
            'DECODER': {
                'HIDDEN_DIMS': [256, 128, 64, 32]
            }
        },
        'CHECKPOINT': {
            'SAVE_DIR': './outputs/stage2_checkpoints'
        }
    }

    try:
        # Create dataset
        print(" Creating dataset...")
        dataset = Stage2Dataset(config, mode="train")
        print(f"[OK] Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS'],
            pin_memory=True
        )

        # Create model
        print(" Creating model...")
        model = Stage2Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[OK] Model created with {total_params:,} parameters")

        # Create trainer
        print(" Setting up trainer...")
        trainer = create_trainer(
            stage='stage2',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None
        )

        # Start training
        print(" Starting training...")
        trainer.train()

        print(" Stage 2 training completed!")
        return True

    except Exception as e:
        print(f" Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print(" ECG Digitization Quick Training")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("  CUDA not available, using CPU (training will be slow)")

    # Check data
    data_path = "../ecg_data/physionet-ecg-image-digitization"
    if os.path.exists(data_path):
        print(f"[OK] Data directory found: {data_path}")
    else:
        print(f" Data directory not found: {data_path}")
        return

    print("\nWhich stage do you want to train?")
    print("1. Stage 0 (Image normalization and keypoint detection)")
    print("2. Stage 1 (Image rectification and grid detection)")
    print("3. Stage 2 (Signal digitization)")
    print("4. Test all stages (quick test)")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        success = quick_train_stage0()
    elif choice == '2':
        success = quick_train_stage1()
    elif choice == '3':
        success = quick_train_stage2()
    elif choice == '4':
        print(" Testing all stages...")
        success1 = quick_train_stage0()
        success2 = quick_train_stage1()
        success3 = quick_train_stage2()
        success = success1 and success2 and success3
    else:
        print(" Invalid choice")
        return

    if success:
        print("\n Training completed successfully!")
        print(" Checkpoints saved in ./outputs/")
    else:
        print("\n Training failed!")


if __name__ == "__main__":
    main()