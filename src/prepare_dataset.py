import torch
import torchaudio
import yaml
import textgrid
from pathlib import Path
from tqdm import tqdm
from symbols import symbol_to_id
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram


def get_mel_spectrogram(audio_path, config):
    """
    Computes Log-Mel Spectrogram using standard Torchaudio.
    Matches standard HiFi-GAN preprocessing.
    """
    # 1. Load Audio
    wav, sr = torchaudio.load(str(audio_path))

    # 2. Resample if needed
    if sr != config["sample_rate"]:
        resampler = torchaudio.transforms.Resample(sr, config["sample_rate"])
        wav = resampler(wav)

    # 3. Create Transform
    # center=False is CRITICAL for TTS alignment.
    # It ensures frame 0 corresponds exactly to time 0.0s.
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

    return spectrogram.squeeze(0)


def seconds_to_frames(seconds, sample_rate, hop_length):
    return int(seconds * sample_rate / hop_length)


def convert_textgrid_to_pt(tg_path, wav_path, config):
    # 1. Get Mel Spectrogram
    mel_spectrogram = get_mel_spectrogram(wav_path, config)
    target_frames = mel_spectrogram.shape[-1]

    # 2. Parse TextGrid
    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        # MFA puts phones in the "phones" tier
        phoneme_tier = tg.getFirst("phones")
    except Exception:
        print(f"Error reading TextGrid: {tg_path}")
        return

    phonemes = []
    durations = []

    # 3. Iterate Intervals
    for interval in phoneme_tier.intervals:
        mark = interval.mark.strip()

        # --- ROBUST SYMBOL MAPPING ---
        # MFA might output: "", "<sil>", "spn", "sil"
        # We want to map ALL of these to our 'sil' ID.
        if mark in ["", "<sil>", "spn", "sil"]:
            if "sil" in symbol_to_id:
                phonemes.append(symbol_to_id["sil"])
            elif "spn" in symbol_to_id:
                phonemes.append(symbol_to_id["spn"])
            else:
                # Should not happen if extraction worked
                continue

        # Valid Phoneme
        elif mark in symbol_to_id:
            phonemes.append(symbol_to_id[mark])

        # Unknown/OOV? Map to silence to be safe
        else:
            if "sil" in symbol_to_id:
                phonemes.append(symbol_to_id["sil"])

        # --- DURATION CALCULATION ---
        duration_sec = interval.maxTime - interval.minTime
        frames = seconds_to_frames(duration_sec, config["sample_rate"], config["hop_length"])
        durations.append(frames)

    # 4. The "Rounding Hack" (Critical)
    # Audio duration vs Sum(Duration) is rarely identical due to rounding.
    # We force the last phoneme to stretch/shrink to match the Mel length.
    total_duration = sum(durations)
    diff = target_frames - total_duration

    if len(durations) > 0:
        durations[-1] += diff

        # Safety: If the last phoneme was tiny and diff was negative,
        # ensure it stays at least 1 frame.
        if durations[-1] < 1:
            durations[-1] = 1
            # If we still have a mismatch, trim the MEL
            new_total = sum(durations)
            if new_total < target_frames:
                mel_spectrogram = mel_spectrogram[:, :new_total]

    # 5. Save
    data = {
        "phonemes": torch.tensor(phonemes, dtype=torch.long),
        "durations": torch.tensor(durations, dtype=torch.long),
        "mel_spectrogram": mel_spectrogram
    }

    save_name = tg_path.with_suffix(".pt").name
    torch.save(data, Path("data/processed") / save_name)


def prepare_dataset():
    root = Path("data")
    wavs_dir = root / "LJSpeech-1.1" / "wavs"
    aligned_dir = root / "aligned"
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = Path("configs") / "data_config.yaml"
    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    print(f"Looking for TextGrids in {aligned_dir}...")
    tg_files = list(aligned_dir.glob("*.TextGrid"))
    print(f"Found {len(tg_files)} files. Processing...")

    for tg_path in tqdm(tg_files):
        file_id = tg_path.stem
        wav_path = wavs_dir / f"{file_id}.wav"

        if not wav_path.exists():
            continue

        try:
            convert_textgrid_to_pt(tg_path, wav_path, data_config)
        except Exception as e:
            # Print error but don't crash the whole loop
            print(f"Failed {file_id}: {e}")


if __name__ == "__main__":
    prepare_dataset()