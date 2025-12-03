import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from data import SweepDataset, SweepEvalDataset, imagenet_transform, train_transform
from model import NEJMbaseline
import warnings

warnings.filterwarnings("ignore")


def rmse_from_mse(mse_tensor):
    return torch.sqrt(mse_tensor + 1e-8)


def train_and_validate(train_csv, val_csv, epochs=100, batch_size=8,
                       n_sweeps_val=8, save_path='checkpoints/best_model.pth'):
    """
    Train and validate the NEJMbaseline model.
    Logs metrics to TensorBoard for both train and validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir="logs")

    # Datasets and loaders
    train_dataset = SweepDataset(train_csv, transform=train_transform, target_frames=16)
    val_dataset = SweepEvalDataset(val_csv, n_sweeps=n_sweeps_val,
                                   transform=imagenet_transform, target_frames=16)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = NEJMbaseline(backbone='resnet18', pretrained=True).to(device)

    criterion = nn.SmoothL1Loss(beta=5.0)  # between L1 and L2
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_mae = float('inf')
    global_step = 0

    for epoch in range(epochs):

        # ---------------- Training ----------------
        model.train()
        train_mae_epoch, train_mse_epoch, train_loss_epoch = 0.0, 0.0, 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for frames, labels in train_pbar:
            frames = frames.to(device, non_blocking=True)           # (B, T, C, H, W)
            labels = labels.float().to(device).unsqueeze(1)         # (B, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs, _ = model(frames)                          # (B, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            with torch.no_grad():
                mae = torch.mean(torch.abs(outputs - labels))
                mse = torch.mean((outputs - labels) ** 2)
                rmse = rmse_from_mse(mse)

            # Logging batch
            writer.add_scalar("Train/Batch_Loss", loss.item(), global_step)
            writer.add_scalar("Train/Batch_MAE", mae.item(), global_step)
            writer.add_scalar("Train/Batch_RMSE", rmse.item(), global_step)
            global_step += 1

            # Running totals
            batch_size_curr = frames.size(0)
            train_loss_epoch += loss.item() * batch_size_curr
            train_mae_epoch += mae.item() * batch_size_curr
            train_mse_epoch += mse.item() * batch_size_curr

            train_pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "mae": f"{mae.item():.3f}",
                "rmse": f"{rmse.item():.3f}"
            })

        # Epoch-level metrics
        num_train = len(train_loader.dataset)
        train_loss_epoch /= num_train
        train_mae_epoch /= num_train
        train_mse_epoch /= num_train
        train_rmse_epoch = (train_mse_epoch ** 0.5)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss_epoch:.4f} "
              f"| MAE: {train_mae_epoch:.4f} | RMSE: {train_rmse_epoch:.4f}")

        writer.add_scalar("Train/Epoch_Loss", train_loss_epoch, epoch + 1)
        writer.add_scalar("Train/Epoch_MAE", train_mae_epoch, epoch + 1)
        writer.add_scalar("Train/Epoch_RMSE", train_rmse_epoch, epoch + 1)

        # ---------------- Validation ----------------
        model.eval()
        val_loss_epoch, val_mae_epoch, val_mse_epoch = 0.0, 0.0, 0.0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)

        with torch.no_grad():
            for sweeps, labels in val_pbar:
                sweeps = sweeps.to(device, non_blocking=True)       # (B, S, T, C, H, W)
                labels = labels.float().to(device).unsqueeze(1)     # (B, 1)

                B, S, T, C, H, W = sweeps.shape
                sweeps = sweeps.view(B * S, T, C, H, W)             # (B*S, T, C, H, W)

                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    outputs, _ = model(sweeps)                      # (B*S, 1)
                    outputs = outputs.view(B, S, 1).mean(dim=1)     # (B, 1) mean over sweeps
                    loss = criterion(outputs, labels)

                mae = torch.mean(torch.abs(outputs - labels))
                mse = torch.mean((outputs - labels) ** 2)
                rmse = rmse_from_mse(mse)

                batch_size_curr = B
                val_loss_epoch += loss.item() * batch_size_curr
                val_mae_epoch += mae.item() * batch_size_curr
                val_mse_epoch += mse.item() * batch_size_curr

                writer.add_scalar("Val/Batch_Loss", loss.item(), global_step)
                writer.add_scalar("Val/Batch_MAE", mae.item(), global_step)
                writer.add_scalar("Val/Batch_RMSE", rmse.item(), global_step)

                val_pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mae": f"{mae.item():.3f}",
                    "rmse": f"{rmse.item():.3f}"
                })

        num_val = len(val_loader.dataset)
        val_loss_epoch /= num_val
        val_mae_epoch /= num_val
        val_mse_epoch /= num_val
        val_rmse_epoch = (val_mse_epoch ** 0.5)

        print(f"Epoch {epoch+1} | Val Loss: {val_loss_epoch:.4f} "
              f"| MAE: {val_mae_epoch:.4f} | RMSE: {val_rmse_epoch:.4f}")

        writer.add_scalar("Val/Epoch_Loss", val_loss_epoch, epoch + 1)
        writer.add_scalar("Val/Epoch_MAE", val_mae_epoch, epoch + 1)
        writer.add_scalar("Val/Epoch_RMSE", val_rmse_epoch, epoch + 1)

        # Step scheduler
        scheduler.step()

        # ---------------- Save best model (by MAE) ----------------
        if val_mae_epoch < best_val_mae:
            best_val_mae = val_mae_epoch
            print(f"✅ Saving new best model (Val MAE: {best_val_mae:.4f})")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_mae': val_mae_epoch
            }, save_path)

    writer.close()


if __name__ == "__main__":
    train_and_validate(
        train_csv="/mnt/Data/hackathon/final_train.csv",
        val_csv="/mnt/Data/hackathon/final_valid.csv",
        epochs=50,
        batch_size=8,
        n_sweeps_val=8,
        save_path="checkpoints/best_model.pth"
    )
