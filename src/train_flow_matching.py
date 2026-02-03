import os

# Allow PyTorch to fall back to CPU for operations missing on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn.functional as F
import tqdm
import time
from pathlib import Path
from dataset import TTSDataset, TTSCollate
from decoder import FlowMatchingDecoder
from utils import load_config
from model import AcousticModel
from length_regulator import LengthRegulator
from ema import ModelEma


MEL_MEAN = -5.521275
MEL_STD = 2.065534


def compute_loss(batch, acoustic_model, flow_decoder, length_regulator, device):
    phonemes = batch["phonemes"].to(device)
    durations = batch["durations"].to(device)
    mel_spectrograms = batch["mel_spectrograms"].to(device)
    mel_lengths = batch["mel_lengths"].to(device)

    # Transpose mel_spectrograms to (Batch, Time, 80)
    mel_spectrograms = mel_spectrograms.transpose(1, 2)

    # Normalize mel spectrograms
    x1 = (mel_spectrograms - MEL_MEAN) / MEL_STD

    # Get encoder outputs from acoustic model
    with torch.no_grad():
        log_duration_preds, encoder_outputs = acoustic_model(phonemes)

    # Length regulation to get aligned encoder outputs
    aligned_encoder_outputs = length_regulator(encoder_outputs, durations)

    # Flow matching training
    batch_size = mel_spectrograms.size(0)

    x0 = torch.randn_like(x1, device=device)  # sample random gaussian noise
    t = torch.rand(batch_size, device=device)  # sample random time steps
    t = t.view(batch_size, 1, 1)  # shape (B, 1, 1)

    x_t = (1 - t) * x0 + t * x1  # shape (B, Mel_Len, 80)

    # Target velocity
    v_target = x1 - x0  # shape (B, Mel_Len, 80)

    # Create mask for valid mel positions
    max_mel_len = mel_spectrograms.size(1)
    mel_mask = torch.arange(max_mel_len, device=device).unsqueeze(0) < mel_lengths.unsqueeze(1)
    mel_mask = mel_mask.float()

    # Forward pass through flow matching decoder
    t_input = t.view(batch_size, 1)  # shape (B, 1)
    v_pred = flow_decoder(x_t, mel_mask, aligned_encoder_outputs, t_input)

    # Compute loss (MSE)
    loss_element_wise = F.mse_loss(v_pred, v_target, reduction='none')
    loss_masked = loss_element_wise * mel_mask.unsqueeze(-1)  # Apply mel mask
    loss = loss_masked.sum() / (mel_mask.sum() * 80)  # Normalize by number of valid elements

    return loss


def train():
    # Seed everything for reproducibility
    torch.manual_seed(42)

    # Create run directory
    os.makedirs("runs", exist_ok=True)
    # Current time for unique run directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"flow_matching_training_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Create subdirectories
    model_checkpoint_dir = run_dir / "model_checkpoints"
    model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_log_dir = run_dir / "training_logs"
    training_log_dir.mkdir(parents=True, exist_ok=True)

    config = load_config("model_config.yaml")
    am_config = config["acoustic_model"]
    fm_config = config["flow_model"]

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
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=fm_config["batch_size"], shuffle=True,
                                                   collate_fn=collate, num_workers=4, pin_memory=True,
                                                   persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=fm_config["batch_size"], shuffle=False,
                                                 collate_fn=collate, num_workers=2, pin_memory=True,
                                                 persistent_workers=True)

    # Load pre-trained acoustic model
    acoustic_model = AcousticModel(
        encoder_dim=am_config["encoder_dim"],
        hidden_dim=am_config["hidden_dim"],
        n_heads=am_config["n_heads"],
        encoder_dropout=am_config["encoder_dropout"],
        encoder_layers=am_config["encoder_layers"],
        duration_predictor_hidden_dim=am_config["duration_predictor_hidden_dim"],
        duration_predictor_dropout=am_config["duration_predictor_dropout"]
    ).to(device)

    acoustic_model.load_state_dict(
        torch.load(am_config["pretrained_model_path"], map_location=device, weights_only=True)
    )
    acoustic_model.eval()  # Set to eval mode
    for param in acoustic_model.parameters():
        param.requires_grad = False  # Freeze parameters

    # Initialize Flow Matching Decoder
    flow_matching_decoder = FlowMatchingDecoder(
        input_dim=fm_config["input_dim"],
        output_dim=fm_config["output_dim"],
        hidden_dim=fm_config["hidden_dim"],
        n_heads=fm_config["n_heads"],
        n_layers=fm_config["n_layers"],
        dropout=fm_config["dropout"]
    ).to(device)

    length_regulator = LengthRegulator().to(device)

    ema_model = ModelEma(flow_matching_decoder, decay=0.9995, device=device)

    optimizer = torch.optim.AdamW(flow_matching_decoder.parameters(), lr=fm_config["learning_rate"])

    scaler = None
    if device.type == 'cuda':
        print("Using mixed precision training with GradScaler.")
        scaler = torch.amp.GradScaler()

    # Logging setup
    print("Starting training...")
    epoch_train_losses = []
    epoch_val_losses = []

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = fm_config.get("patience", 10)

    # Training Loop
    for epoch in range(fm_config["epochs"]):
        running_loss = 0.0
        pbar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{fm_config['epochs']}")

        flow_matching_decoder.train()
        for batch in pbar:
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    loss = compute_loss(batch, acoustic_model, flow_matching_decoder, length_regulator, device)
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_matching_decoder.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss(batch, acoustic_model, flow_matching_decoder, length_regulator, device)
                loss.backward()
                optimizer.step()

            ema_model.update(flow_matching_decoder)

            running_loss += loss.item()
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_dataloader)
        epoch_train_losses.append(avg_loss)

        # Validation loop
        flow_matching_decoder.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                if scaler is not None:
                    with torch.amp.autocast(device_type=device.type):
                        loss = compute_loss(val_batch, acoustic_model, flow_matching_decoder, length_regulator, device)
                else:
                    loss = compute_loss(val_batch, acoustic_model, flow_matching_decoder, length_regulator, device)

                val_running_loss += loss.item()

        val_avg_loss = val_running_loss / len(val_dataloader)
        epoch_val_losses.append(val_avg_loss)
        print(f"Epoch {epoch + 1} completed. Train Loss: {avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")

        # Early stopping check
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            patience_counter = 0
            # Save the best model
            checkpoint_path = model_checkpoint_dir / "flow_matching_decoder_best.pt"
            torch.save(flow_matching_decoder.state_dict(), checkpoint_path)
            torch.save(ema_model.module.state_dict(), model_checkpoint_dir / "flow_matching_decoder_ema_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = model_checkpoint_dir / f"flow_matching_decoder_epoch_{epoch + 1}.pt"
            torch.save(flow_matching_decoder.state_dict(), checkpoint_path)

    # Save final model
    final_checkpoint_path = model_checkpoint_dir / "flow_matching_decoder_final.pt"
    torch.save(flow_matching_decoder.state_dict(), final_checkpoint_path)

    # Save training log
    log_path = training_log_dir / "training_log.txt"
    with open(log_path, "w") as f:
        for epoch_idx, (train_loss, val_loss) in enumerate(zip(epoch_train_losses, epoch_val_losses), 1):
            f.write(f"Epoch {epoch_idx}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")


if __name__ == "__main__":
    train()