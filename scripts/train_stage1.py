"""Direct training script for Stage 1 - Image Rectification and Grid Detection."""

import os
import sys
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def print_with_time(message):
    """Print message with timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def main():
    """Main training function."""
    print("ECG Digitization - Stage 1 Training")
    print("=" * 50)
    print("Stage 1: Image Rectification and Grid Detection")

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

    # Configuration for Stage 1
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
            'LEARNING_RATE': 1e-4,
            'GRADIENT_CLIP': 1.0
        },
        'MODEL': {
            'INPUT_SIZE': [1152, 1440],
            'BACKBONE': {
                'NAME': 'resnet34',
                'PRETRAINED': False  # Faster loading
            },
            'GRID_CONFIG': {
                'H_LINES': 44,
                'V_LINES': 57,
                'MV_TO_PIXEL': 79.0,
                'ZERO_MV': [703.5, 987.5, 1271.5, 1531.5]
            }
        },
        'DATA': {
            'NORMALIZE': {
                'MEAN': [0.485, 0.456, 0.406],
                'STD': [0.229, 0.224, 0.225]
            }
        },
        'CHECKPOINT': {
            'SAVE_DIR': './outputs/stage1_checkpoints',
            'SAVE_INTERVAL': 1
        }
    }

    try:
        # Import required modules - test individual imports
        print_with_time("Testing imports...")
        import data.dataset
        print_with_time("data.dataset imported successfully")
        import models
        print_with_time("models imported successfully")
        from data.dataset import Stage1Dataset
        from models import Stage1Net
        print_with_time("Stage1 imports successful")

        # Create checkpoint directory
        checkpoint_dir = config['CHECKPOINT']['SAVE_DIR']
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_with_time(f"Checkpoint directory: {checkpoint_dir}")

        # Create dataset
        print_with_time("Creating Stage 1 dataset...")
        dataset = Stage1Dataset(config, mode="train")
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

        print_with_time(f"Data loader created with batch size {config['TRAIN']['BATCH_SIZE']}")

        # Create model
        print_with_time("Creating Stage 1 model...")
        device = torch.device(config['DEVICE']['DEVICE'])
        model = Stage1Net(config)
        model = model.to(device)
        print_with_time("Stage 1 model created successfully")

        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_with_time(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=config['TRAIN']['LEARNING_RATE'])
        criterion = nn.MSELoss()  # Basic loss for grid detection
        print_with_time("Optimizer and loss function created")

        # Training loop
        print_with_time("Starting training...")
        model.train()

        for epoch in range(config['TRAIN']['EPOCHS']):
            epoch_loss = 0.0
            num_batches = 0

            print_with_time(f"Epoch {epoch+1}/{config['TRAIN']['EPOCHS']}")

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                elif isinstance(batch, (list, tuple)):
                    batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
                else:
                    batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch

                try:
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(batch)

                    # Simple loss calculation - adapt based on actual model output format
                    if isinstance(outputs, dict):
                        if 'grid_points' in outputs and 'grid_points' in batch:
                            loss = criterion(outputs['grid_points'], batch['grid_points'])
                        elif 'heatmap' in outputs and 'heatmap' in batch:
                            loss = criterion(outputs['heatmap'], batch['heatmap'])
                        else:
                            # Fallback: use first tensor output
                            output_tensor = list(outputs.values())[0]
                            if isinstance(output_tensor, torch.Tensor):
                                target_tensor = batch.get('target', batch.get('grid_points',
                                                                                 list(batch.values())[0]))
                                loss = criterion(output_tensor, target_tensor)
                            else:
                                continue
                    else:
                        # Simple tensor output
                        target = batch.get('target', batch.get('grid_points',
                                                              list(batch.values())[0] if batch else outputs))
                        loss = criterion(outputs, target)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    if config['TRAIN'].get('GRADIENT_CLIP', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['TRAIN']['GRADIENT_CLIP'])

                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 10 == 0:
                        print_with_time(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue

            # Calculate epoch statistics
            avg_loss = epoch_loss / max(num_batches, 1)
            print_with_time(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % config['CHECKPOINT']['SAVE_INTERVAL'] == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"stage1_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': config
                }, checkpoint_path)
                print_with_time(f"Checkpoint saved: {checkpoint_path}")

        # Final model save
        final_path = os.path.join(checkpoint_dir, "stage1_final.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, final_path)
        print_with_time(f"Final model saved: {final_path}")

        print_with_time("Stage 1 training completed successfully!")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()