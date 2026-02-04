import os

# Allow PyTorch to fall back to CPU for operations missing on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pathlib import Path
import torch
import torch.nn.functional as F
import tqdm
import time
from dataset import TTSDataset, TTSCollate
from model import AcousticModel
from utils import load_config
from symbols import symbol_to_id


def train():
    # Seed everything for reproducibility
    torch.manual_seed(42)

    # Create run directory
    os.makedirs("runs", exist_ok=True)
    # Current time for unique run directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"duration_training_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Create subdirectories
    model_checkpoint_dir = run_dir / "model_checkpoints"
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log_dir = run_dir / "training_logs"
    training_log_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("model_config.yaml")["acoustic_model"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    dataset = TTSDataset("data/processed")
    collate = TTSCollate()

    # Create train and validation dataloaders
    dataset_size = len(dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                                   collate_fn=collate, num_workers=4, pin_memory=True,
                                                   persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                                                 collate_fn=collate, num_workers=2, pin_memory=True,
                                                 persistent_workers=True)

    model = AcousticModel(
        encoder_dim=config["encoder_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        encoder_dropout=config["encoder_dropout"],
        encoder_layers=config["encoder_layers"],
        duration_predictor_hidden_dim=config["duration_predictor_hidden_dim"],
        duration_predictor_dropout=config["duration_predictor_dropout"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    PAD_ID = symbol_to_id["pad"]

    # Logging setup
    print("Starting training...")
    epoch_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.get("patience", 10)

    # Training Loop
    for epoch in range(config["epochs"]):
        running_loss = 0.0
        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['epochs']}")

        model.train()
        for batch in pbar:
            phonemes = batch["phonemes"].to(device)
            durations = batch["durations"].to(device)

            optimizer.zero_grad()

            # Forward pass
            log_duration_preds, _ = model(phonemes)

            # Compute loss (only consider non-padded lengths)
            mask = (phonemes != PAD_ID).float()  # Mask for non-padded tokens

            log_duration_targets = torch.log1p(durations.float())  # log(1 + duration)
            loss_unreduced = F.huber_loss(log_duration_preds, log_duration_targets, reduction='none')

            # Ensure padding errors do not contribute to loss
            loss_masked = loss_unreduced * mask

            # Normalize by number of valid (non-padded) elements
            loss = loss_masked.sum() / mask.sum()

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_dataloader)
        epoch_losses.append(avg_loss)

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                phonemes = val_batch["phonemes"].to(device)
                durations = val_batch["durations"].to(device)

                log_duration_preds, _ = model(phonemes)

                mask = (phonemes != PAD_ID).float()
                log_duration_targets = torch.log1p(durations.float())

                loss_unreduced = F.huber_loss(log_duration_preds, log_duration_targets, reduction='none')
                loss_masked = loss_unreduced * mask
                loss = loss_masked.sum() / mask.sum()

                val_running_loss += loss.item()

        val_avg_loss = val_running_loss / len(val_dataloader)

        # Epoch completed
        print(
            f"Epoch [{epoch + 1}/{config['epochs']}] finished | Train Loss: {avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")

        # Early stopping check
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_checkpoint_dir / "duration_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered")
                break

        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_checkpoint_dir / f"duration_model_epoch_{epoch + 1}.pth")

    # Save the trained model
    torch.save(model.state_dict(), model_checkpoint_dir / "duration_model_final.pth")

    # Save training log
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(training_log_dir / f"training_log_{timestamp}.txt", "w") as f:
        for epoch, loss in enumerate(epoch_losses, 1):
            f.write(f"Epoch {epoch}: Loss = {loss:.4f}\n")


if __name__ == "__main__":
    train()
