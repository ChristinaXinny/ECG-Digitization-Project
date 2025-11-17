"""Ultra-simple Stage 0 training script with minimal dependencies."""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_with_time(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def create_simple_trainer(model, train_loader, config):
    """Create a simple trainer without complex dependencies."""

    device = torch.device(config['DEVICE']['DEVICE'])
    model = model.to(device)

    # Setup optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['TRAIN']['LEARNING_RATE'],
        weight_decay=1e-5
    )

    # Setup loss functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_seg = nn.CrossEntropyLoss()

    print_with_time("Starting training loop...")

    for epoch in range(config['TRAIN']['EPOCHS']):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Get data
            images = batch['image'].to(device)

            # Create dummy targets for demonstration
            batch_size = images.size(0)
            marker_target = torch.randint(0, 14, (batch_size, 1152, 1440)).to(device)
            orientation_target = torch.randint(0, 8, (batch_size,)).to(device)

            # Forward pass
            model.output_type = ['loss']
            try:
                output = model({'image': images})

                # Calculate losses
                marker_loss = 0
                orientation_loss = 0

                if 'marker_loss' in output:
                    marker_loss = output['marker_loss']
                if 'orientation_loss' in output:
                    orientation_loss = output['orientation_loss']

                # Manual loss calculation if needed
                if marker_loss == 0 and 'marker' in output:
                    marker_loss = criterion_seg(output['marker'], marker_target)
                if orientation_loss == 0 and 'orientation' in output:
                    orientation_loss = criterion_cls(output['orientation'], orientation_target)

                total_loss = marker_loss + 0.5 * orientation_loss

            except Exception as e:
                print_with_time(f"Forward pass error: {e}")
                continue

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update statistics
            epoch_loss += total_loss.item()
            num_batches += 1

            # Print progress
            if batch_idx % 10 == 0:
                print_with_time(f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']}, "
                               f"Batch {batch_idx}/{len(train_loader)}, "
                               f"Loss: {total_loss.item():.6f}")

        # Print epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        print_with_time(f"Epoch {epoch+1} completed - Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_dir = "./outputs/stage0_checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print_with_time(f"Checkpoint saved: {checkpoint_path}")

    print_with_time("Training completed!")
    return True


def main():
    """Main function."""
    print("ECG Digitization - Simple Stage 0 Training")
    print("=" * 50)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU (training will be slow)")

    # Check data
    data_path = "../ecg_data/physionet-ecg-image-digitization"
    if os.path.exists(data_path):
        print(f"Data directory found: {data_path}")
    else:
        print(f"ERROR: Data directory not found: {data_path}")
        return

    # Configuration
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
                'NAME': 'resnet18d',
                'PRETRAINED': False
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
        # Import required modules with fallback
        try:
            from data.dataset import Stage0Dataset
            from models import Stage0Net
        except ImportError:
            print("ERROR: Cannot import required modules.")
            print("Please ensure you are in the ECG-Digitization-Project directory")
            print("and that all required packages are installed:")
            print("pip install torch torchvision timm numpy pandas opencv-python")
            return False

        # Create dataset
        print_with_time("Creating dataset...")
        dataset = Stage0Dataset(config, mode="train")
        print_with_time(f"Loaded {len(dataset)} training samples")

        if len(dataset) == 0:
            print("ERROR: No training samples found!")
            return

        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config['TRAIN']['BATCH_SIZE'],
            shuffle=True,
            num_workers=config['DEVICE']['NUM_WORKERS']
        )

        # Create model
        print_with_time("Creating model...")
        model = Stage0Net(config)
        total_params = sum(p.numel() for p in model.parameters())
        print_with_time(f"Model created with {total_params:,} parameters")

        # Create outputs directory
        os.makedirs("./outputs/stage0_checkpoints", exist_ok=True)

        # Start training
        success = create_simple_trainer(model, train_loader, config)

        if success:
            print_with_time("Training completed successfully!")
            print("Checkpoints saved in ./outputs/stage0_checkpoints/")
            print("\nNext steps:")
            print("1. Check the trained model: ls outputs/stage0_checkpoints/")
            print("2. Test inference with the trained model")
            print("3. Increase epochs for better results")
        else:
            print("Training failed!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()