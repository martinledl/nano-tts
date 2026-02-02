import torch
import torchaudio
import yaml
import textgrid
from pathlib import Path
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

# Add the parent directory (root) to sys.path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import *
from src.symbols import *


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


def convert_textgrid_to_pt(file_path, root_path, config):
    # Construct paths using Pathlib
    # file_path is the TextGrid path
    audio_filename = file_path.with_suffix(".wav").name
    audio_path = root_path / DATASET_DIR / "wavs" / audio_filename

    # Get Mel Spectrogram
    # Shape: [Mels, Time]
    mel_spectrogram = get_mel_spectrogram(audio_path, config)
    target_frames = mel_spectrogram.shape[-1]  # Time is now correctly the last dim

    tg = textgrid.TextGrid.fromFile(str(file_path))
    phoneme_tier = tg.getFirst("phones")

    phonemes = []
    durations = []

    for interval in phoneme_tier.intervals:
        phoneme = interval.mark

        # MFA uses empty strings or <sil> for silence.
        # Ensure your get_mfa_symbol_id handles "" correctly!
        phonemes.append(get_mfa_symbol_id(phoneme))

        duration_sec = interval.maxTime - interval.minTime
        frames = seconds_to_frames(duration_sec, sample_rate=config["sample_rate"],
                                   hop_length=config["hop_length"])
        durations.append(frames)

    # Fix Rounding Errors
    total_duration = sum(durations)
    diff = target_frames - total_duration

    if len(durations) > 0:
        durations[-1] += diff

        if durations[-1] < 0:
            # This is a critical error, usually means G2P/Alignment failed massively
            print(f"CRITICAL: Negative duration in {file_path.name}. Skipping.")
            return

    data = {
        "phonemes": torch.tensor(phonemes, dtype=torch.long),
        "durations": torch.tensor(durations, dtype=torch.long),
        "mel_spectrogram": mel_spectrogram
    }

    # Save to processed folder
    save_name = file_path.with_suffix(".pt").name
    torch.save(data, root_path / "data" / "processed" / save_name)


def prepare_dataset(root_path_str, output_path_str):
    root_path = Path(root_path_str)
    output_path = Path(output_path_str)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("configs") / "data_config.yaml"
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    # Path to alignments
    alignment_dir = root_path / ALIGNMENT_DIR

    print(alignment_dir)

    # Use glob for cleaner iteration
    files = list(alignment_dir.glob("*.TextGrid"))
    print(f"Found {len(files)} alignment files.")

    for tg_path in files:
        convert_textgrid_to_pt(tg_path, root_path, data_config)


if __name__ == "__main__":
    ROOT = "./"
    OUTPUT = "data/processed"

    prepare_dataset(ROOT, OUTPUT)