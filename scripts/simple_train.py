"""Simple training script for ECG digitization - minimal dependencies."""

import os
import sys
import torch
import time
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_with_time(message):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def train_stage0():
    """Simple training for Stage 0 model."""
    print_with_time("🚀 Starting Stage 0 Training...")

    # Configuration
    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 0  # Simplified
        },
        'TRAIN': {
            'BATCH_SIZE': 2,  # Small batch size
            'EPOCHS': 3,      # Few epochs for quick test
            'LEARNING_RATE': 1e-4
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet18d',
                'PRETRAINED': False  # Faster loading
            }
        },
        'DATA': {
            'NORMALIZE': {
                'MEAN': [0.485, 0.456, 0.406],
                'STD': [0.229, 0.224, 0.225]
            }
        }
    }

    try:
        # Import here to handle dependency issues
        from data.dataset import Stage0Dataset
        from models import Stage0Net
        from engines.stage_trainer import create_trainer

        # Create dataset
        print_with_time("📊 Creating dataset...")
        dataset = Stage0Dataset(config, mode="train")
        print_with_time(f"✅ Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS']
        )

        # Create model
        print_with_time("🧠 Creating model...")
        model = Stage0Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print_with_time(f"✅ Model created with {total_params:,} parameters")

        # Create trainer
        print_with_time("🏃‍♂️ Setting up trainer...")
        trainer = create_trainer(
            stage='stage0',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None
        )

        # Start training
        print_with_time("🎯 Starting training...")
        trainer.train()

        print_with_time("🎉 Stage 0 training completed!")
        return True

    except ImportError as e:
        print_with_time(f"❌ Import error: {e}")
        print_with_time("Please install required packages: pip install -r requirements_minimal.txt")
        return False
    except Exception as e:
        print_with_time(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_stage1():
    """Simple training for Stage 1 model."""
    print_with_time("🚀 Starting Stage 1 Training...")

    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 0
        },
        'TRAIN': {
            'BATCH_SIZE': 2,
            'EPOCHS': 3,
            'LEARNING_RATE': 1e-4
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet34',
                'PRETRAINED': False
            }
        },
        'DATA': {
            'GRID_CONFIG': {
                'H_LINES': 44,
                'V_LINES': 57
            }
        }
    }

    try:
        from data.dataset import Stage1Dataset
        from models import Stage1Net
        from engines.stage_trainer import create_trainer

        # Create dataset
        print_with_time("📊 Creating dataset...")
        dataset = Stage1Dataset(config, mode="train")
        print_with_time(f"✅ Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS']
        )

        # Create model
        print_with_time("🧠 Creating model...")
        model = Stage1Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print_with_time(f"✅ Model created with {total_params:,} parameters")

        # Create trainer
        print_with_time("🏃‍♂️ Setting up trainer...")
        trainer = create_trainer(
            stage='stage1',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None
        )

        # Start training
        print_with_time("🎯 Starting training...")
        trainer.train()

        print_with_time("🎉 Stage 1 training completed!")
        return True

    except Exception as e:
        print_with_time(f"❌ Training failed: {e}")
        return False


def train_stage2():
    """Simple training for Stage 2 model."""
    print_with_time("🚀 Starting Stage 2 Training...")

    config = {
        'COMPETITION': {
            'MODE': 'local',
            'KAGGLE_DIR': '../ecg_data/physionet-ecg-image-digitization'
        },
        'DEVICE': {
            'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
            'NUM_WORKERS': 0
        },
        'TRAIN': {
            'BATCH_SIZE': 1,  # Larger images, smaller batch
            'EPOCHS': 3,
            'LEARNING_RATE': 1e-4
        },
        'MODEL': {
            'INPUT_SIZE': [1696, 2176],
            'BACKBONE': {
                'NAME': 'resnet34',
                'PRETRAINED': False
            }
        }
    }

    try:
        from data.dataset import Stage2Dataset
        from models import Stage2Net
        from engines.stage_trainer import create_trainer

        # Create dataset
        print_with_time("📊 Creating dataset...")
        dataset = Stage2Dataset(config, mode="train")
        print_with_time(f"✅ Loaded {len(dataset)} training samples")

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS']
        )

        # Create model
        print_with_time("🧠 Creating model...")
        model = Stage2Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print_with_time(f"✅ Model created with {total_params:,} parameters")

        # Create trainer
        print_with_time("🏃‍♂️ Setting up trainer...")
        trainer = create_trainer(
            stage='stage2',
            model=model,
            config=config,
            train_dataloader=train_loader,
            val_dataloader=None
        )

        # Start training
        print_with_time("🎯 Starting training...")
        trainer.train()

        print_with_time("🎉 Stage 2 training completed!")
        return True

    except Exception as e:
        print_with_time(f"❌ Training failed: {e}")
        return False


def main():
    """Main function."""
    print("🔥 ECG Digitization Simple Training")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA not available, using CPU (training will be slow)")

    # Check data
    data_path = "../ecg_data/physionet-ecg-image-digitization"
    if os.path.exists(data_path):
        print(f"✅ Data directory found: {data_path}")
    else:
        print(f"❌ Data directory not found: {data_path}")
        print("Please make sure your ecg_data directory is in the correct location")
        return

    print("\nWhich stage do you want to train?")
    print("1. Stage 0 (Image normalization and keypoint detection)")
    print("2. Stage 1 (Image rectification and grid detection)")
    print("3. Stage 2 (Signal digitization)")

    try:
        choice = input("\nEnter your choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\n\nTraining cancelled.")
        return

    if choice == '1':
        success = train_stage0()
    elif choice == '2':
        success = train_stage1()
    elif choice == '3':
        success = train_stage2()
    else:
        print("❌ Invalid choice")
        return

    if success:
        print("\n🎉 Training completed successfully!")
        print("📁 Checkpoints saved in ./outputs/")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install -r requirements_minimal.txt")
        print("2. Run with more epochs for better results")
        print("3. Use the trained models for inference")
    else:
        print("\n❌ Training failed!")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main()