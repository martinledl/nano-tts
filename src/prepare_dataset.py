import torch
import torchaudio
import yaml
import textgrid
from pathlib import Path
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
from tqdm import tqdm

# Add the parent directory (root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from symbols import *


def seconds_to_frames(seconds, sample_rate, hop_length):
    frames = int(seconds * sample_rate / hop_length)
    return frames


def get_mel_spectrogram(audio_path, config):
    """
    Computes Mel Spectrogram using SpeechBrain implementation to match HiFi-GAN.
    Returns: Tensor of shape [Mels, Time]
    """
    # Load Audio
    # torchaudio loads as [channels, time]
    wav, sr = torchaudio.load(str(audio_path))

    # Resample if needed
    if sr != config["sample_rate"]:
        resampler = torchaudio.transforms.Resample(sr, config["sample_rate"])
        wav = resampler(wav)

    # Use SpeechBrain's Exact Implementation
    # Note: wav.squeeze(0) makes it 1D [Time]. SB treats this as a batch of 1.
    spectrogram, _ = mel_spectogram(
        audio=wav.squeeze(0),
        sample_rate=config["sample_rate"],
        hop_length=config["hop_length"],
        win_length=None,
        n_mels=config["n_mels"],
        n_fft=config["n_fft"],
        f_min=0.0,
        f_max=8000.0,
        power=1,
        normalized=False,
        min_max_energy_norm=True,
        norm="slaney",
        mel_scale="slaney",
        compression=True
    )

    spectrogram = spectrogram.squeeze(0)
    return spectrogram


def convert_textgrid_to_pt(tg_path, wav_path, config):
    # 1. Get Mel Spectrogram
    mel_spectrogram = get_mel_spectrogram(wav_path, config)
    target_frames = mel_spectrogram.shape[-1]

    # 2. Parse TextGrid (MFA Output)
    tg = textgrid.TextGrid.fromFile(str(tg_path))
    # MFA usually puts phones in the "phones" tier (tier 1 usually)
    # Depending on MFA version, it might be named differently.
    # We try getting the tier by name "phones", or fall back to index 1.
    try:
        phoneme_tier = tg.getFirst("phones")
    except ValueError:
        # Fallback: assume it's the second tier (IntervalTier)
        phoneme_tier = tg[1]

    phonemes = []
    durations = []

    for interval in phoneme_tier.intervals:
        mark = interval.mark.strip()

        # 3. Handle Special Tokens
        # MFA marks silence as "" (empty), "<sil>", or "spn"
        if mark == "" or mark == "<sil>" or mark == "spn" or mark == "sil":
            if "spn" in symbol_to_id:
                phonemes.append(symbol_to_id["spn"])
            else:
                # Fallback if spn missing (shouldn't happen with your extraction)
                continue
        elif mark in symbol_to_id:
            phonemes.append(symbol_to_id[mark])
        else:
            # Unknown symbol? (e.g. noise marker)
            # Map to spn (silence) to be safe
            phonemes.append(symbol_to_id["spn"])

        # 4. Calculate Duration
        duration_sec = interval.maxTime - interval.minTime
        frames = seconds_to_frames(duration_sec, sample_rate=config["sample_rate"],
                                   hop_length=config["hop_length"])
        durations.append(frames)

    # 5. Fix Rounding Errors (CRITICAL)
    # The sum of durations MUST equal the mel spectrogram length exactly.
    total_duration = sum(durations)
    diff = target_frames - total_duration

    if len(durations) > 0:
        durations[-1] += diff

        # If correction made the last duration negative (rare, but possible with short files)
        if durations[-1] < 1:
            durations[-1] = 1
            # Propagate error backward or resize mel?
            # Easiest: Force Resize Mel to match sum(durations)
            actual_len = sum(durations)
            if actual_len != target_frames:
                mel_spectrogram = mel_spectrogram[:, :actual_len]

    data = {
        "phonemes": torch.tensor(phonemes, dtype=torch.long),
        "durations": torch.tensor(durations, dtype=torch.long),
        "mel_spectrogram": mel_spectrogram
    }

    # Save to processed folder
    save_name = tg_path.with_suffix(".pt").name
    torch.save(data, Path("data") / "processed" / save_name)


def prepare_dataset(root_path_str, output_path_str):
    root = Path(root_path_str)
    wavs_dir = root / "LJSpeech-1.1" / "wavs"
    aligned_dir = root / "aligned"  # Output folder from MFA
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("configs") / "data_config.yaml"
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    print(f"Looking for TextGrids in {aligned_dir}...")
    tg_files = list(aligned_dir.glob("*.TextGrid"))
    print(f"Found {len(tg_files)} files.")

    for tg_path in tqdm(tg_files):
        file_id = tg_path.stem
        wav_path = wavs_dir / f"{file_id}.wav"
        output_path = processed_dir / f"{file_id}.pt"

        if not wav_path.exists():
            print(f"Missing WAV for {file_id}, skipping.")
            continue

        try:
            convert_textgrid_to_pt(tg_path, wav_path, data_config)
        except Exception as e:
            print(f"Error processing {file_id}: {e}")


if __name__ == "__main__":
    ROOT = "./data/"
    OUTPUT = "data/processed"

    prepare_dataset(ROOT, OUTPUT)