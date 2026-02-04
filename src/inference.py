import os

# Allow PyTorch to fall back to CPU for operations missing on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import yaml
import torch
import torchaudio
import numpy as np
from speechbrain.inference.vocoders import HIFIGAN
from model import AcousticModel
from decoder import FlowMatchingDecoder
from length_regulator import LengthRegulator
from text_processing import text_to_sequence

# --- CONFIGURATION ---
SPN_ID = 47  # The ID for silence/space in your symbol set
SILENCE_MEL_VAL = -11.5  # The value of silence in Log-Mel space

MEL_MEAN = -5.521275
MEL_STD = 2.065534

# Initialize Vocoder Globally
# It will download automatically to 'pretrained_models/' on the first run.
hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="pretrained_models/hifigan-ljspeech",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_models(config_path, am_path, fm_path, device):
    config = load_config(config_path)

    # 1. Load Acoustic Model
    am_conf = config['acoustic_model']
    acoustic_model = AcousticModel(
        encoder_dim=am_conf['encoder_dim'],
        hidden_dim=am_conf['hidden_dim'],
        n_heads=am_conf['n_heads'],
        encoder_layers=am_conf['encoder_layers'],
        encoder_dropout=am_conf['encoder_dropout'],
        duration_predictor_hidden_dim=am_conf['duration_predictor_hidden_dim'],
        duration_predictor_dropout=am_conf['duration_predictor_dropout']
    ).to(device)

    # 2. Load Flow Matching Decoder
    # Handle key naming variation
    fm_conf = config.get('flow_matching_decoder', config.get('flow_model'))

    decoder = FlowMatchingDecoder(
        input_dim=fm_conf['input_dim'],
        output_dim=fm_conf['output_dim'],
        hidden_dim=fm_conf['hidden_dim'],
        n_layers=fm_conf['n_layers'],
        n_heads=fm_conf['n_heads'],
        dropout=fm_conf['dropout']
    ).to(device)

    # 3. Load Weights
    print(f"Loading Acoustic Model from: {am_path}")
    acoustic_model.load_state_dict(torch.load(am_path, map_location=device, weights_only=True))

    print(f"Loading Flow Decoder from: {fm_path}")
    decoder.load_state_dict(torch.load(fm_path, map_location=device, weights_only=True))

    acoustic_model.eval()
    decoder.eval()

    return acoustic_model, decoder


def save_audio_hifigan(mel_spec, filename, sample_rate=22050):
    """
    Robust wrapper for HiFi-GAN inference.
    Handles dimension mismatches for both input (Mel) and output (Audio).
    """
    # 1. Sanitize Input (Mel Spectrogram)
    # We need [Batch, Mel, Time] -> [1, 80, T]
    if mel_spec.dim() == 2:
        # If [Time, 80] or [80, Time], we need to check which is which
        if mel_spec.size(0) == 80:
            mel_spec = mel_spec.unsqueeze(0)  # -> [1, 80, T]
        else:
            mel_spec = mel_spec.transpose(0, 1).unsqueeze(0)  # -> [1, 80, T]
    elif mel_spec.dim() == 3:
        # If [Batch, Time, 80], transpose to [Batch, 80, Time]
        if mel_spec.size(2) == 80:
            mel_spec = mel_spec.transpose(1, 2)

    # 2. Run Vocoder
    with torch.no_grad():
        # returns [Batch, Time] or [Batch, 1, Time] depending on version
        waveform = hifi_gan.decode_batch(mel_spec)

    # 3. Sanitize Output (Waveform)
    # torchaudio.save expects [Channels, Time] (2D)
    waveform = waveform.squeeze()  # Flatten everything -> [Time]
    waveform = waveform.unsqueeze(0)  # Add channel dim -> [1, Time]

    # 4. Save
    torchaudio.save(filename, waveform.cpu(), sample_rate)
    print(f"HiFi-GAN Audio saved to {filename}")


def infer(text, am_checkpoint, fm_checkpoint, config_path, output_path, steps=50, device='cuda'):
    print(f"Generating for text: '{text}'")

    am, decoder = load_models(config_path, am_checkpoint, fm_checkpoint, device)
    length_regulator = LengthRegulator().to(device)

    # 1. Text to Phonemes
    sequence = text_to_sequence(text, ["english_cleaners"])

    # === FIX 1: APPEND SILENCE TOKEN ===
    # This prevents the attention mechanism from corrupting the last word.
    sequence.append(SPN_ID)
    # ===================================

    phonemes = torch.tensor([sequence]).to(device)  # [1, Seq_Len]

    with torch.no_grad():
        # 2. Get Duration & Text Encoding
        log_durations, encoder_outputs = am(phonemes)

        # Convert log durations to integers
        durations = torch.exp(log_durations) - 1
        durations = torch.clamp(durations, min=1).round().long()

        print(f"Predicted Duration (Frames): {durations.sum().item()}")

        # 3. Align Text
        aligned_text = length_regulator(encoder_outputs, durations)  # [1, Total_Len, Dim]

        # 4. Flow Matching (ODE Solver)
        # Start from Random Noise (Normal Distribution)
        batch_size, seq_len, _ = aligned_text.shape
        x = torch.randn(batch_size, seq_len, 80).to(device)

        # Padding Mask (All ones because we are generating full length)
        mask = torch.ones(batch_size, seq_len).to(device)

        dt = 1.0 / steps

        print("Running ODE Solver...")
        for i in range(steps):
            t_scalar = i / steps
            t = torch.tensor([t_scalar], device=device).view(batch_size, 1)

            # Predict Velocity
            v_pred = decoder(x, mask, aligned_text, t)

            # Euler Step: x_new = x_old + velocity * dt
            x = x + v_pred * dt

        # Denormalize Mel Spectrogram
        x = x * MEL_STD + MEL_MEAN

        # Add ~200ms of silence so HiFi-GAN doesn't crop the last word.
        n_pad = 20
        # Create silence tensor [Batch, n_pad, 80]
        padding = torch.full((batch_size, n_pad, 80), SILENCE_MEL_VAL, device=device)
        x = torch.cat([x, padding], dim=1)

        # 5. Vocode & Save
        # x is [Batch, Time, 80], save_audio_hifigan handles the rest
        save_audio_hifigan(x, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to speak")
    parser.add_argument("--am_ckpt", type=str, required=True, help="Path to Acoustic Model checkpoint")
    parser.add_argument("--fm_ckpt", type=str, required=True, help="Path to Flow Decoder checkpoint")
    parser.add_argument("--config", type=str, default="model_config.yaml", help="Path to model config")
    parser.add_argument("--output", type=str, default="output.wav", help="Output filename")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE steps")

    args = parser.parse_args()

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Inference using device: {device}")

    infer(args.text, args.am_ckpt, args.fm_ckpt, args.config, args.output, args.steps, device)