"""Direct training script for Stage 0 - no interactive input required."""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# Add project directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_with_time(message):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def main():
    """Main training function."""
    import time

    print("ECG Digitization - Stage 0 Training")
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
            'EPOCHS': 5,  # Increase for better training
            'LEARNING_RATE': 1e-4
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet18',
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
        # Import required modules - test individual imports
        print_with_time("Testing imports...")
        import data.dataset
        print_with_time("data.dataset imported successfully")
        import models
        print_with_time("models imported successfully")
        from data.dataset import Stage0Dataset
        from models import Stage0Net
        print_with_time("Basic imports successful")

        # Create a simple trainer instead of using the complex engines
        print_with_time("Using simple training loop (bypassing engines module)...")

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

        # Simple training loop
        print_with_time("Starting simple training loop...")
        device = torch.device(config['DEVICE']['DEVICE'])
        model = model.to(device)

        # Setup optimizer
        import torch.optim as optim
        optimizer = optim.AdamW(model.parameters(), lr=config['TRAIN']['LEARNING_RATE'])

        # Create output directory for checkpoints
        checkpoint_dir = "./outputs/stage0_checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_with_time(f"Created checkpoint directory: {checkpoint_dir}")

        # Training loop
        epochs = config['TRAIN']['EPOCHS']
        best_loss = float('inf')

        for epoch in range(epochs):
            print_with_time(f"Epoch {epoch+1}/{epochs}")
            model.train()

            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch)

                # Compute loss
                if 'total_loss' in outputs:
                    loss = outputs['total_loss']
                else:
                    # Simple MSE loss as fallback
                    loss = torch.tensor(0.0, device=device)
                    for key, value in outputs.items():
                        if 'loss' in key and isinstance(value, torch.Tensor):
                            loss += value

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 10 == 0:
                    print_with_time(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

            avg_loss = epoch_loss / max(num_batches, 1)
            print_with_time(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"stage0_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print_with_time(f"Checkpoint saved: {checkpoint_path}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_path = os.path.join(checkpoint_dir, "stage0_best.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': config
                }, best_model_path)
                print_with_time(f"New best model saved with loss: {avg_loss:.4f}")

        # Save final model
        final_model_path = os.path.join(checkpoint_dir, "stage0_final.pth")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': config
        }, final_model_path)

        print_with_time("Stage 0 training completed successfully!")
        print_with_time(f"Checkpoints saved in {checkpoint_dir}")
        print_with_time("Files saved:")
        print_with_time(f"  - Final model: stage0_final.pth")
        print_with_time(f"  - Best model: stage0_best.pth (loss: {best_loss:.4f})")
        print_with_time(f"  - Epoch checkpoints: stage0_epoch_*.pth")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision timm numpy pandas opencv-python")
        return False
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()