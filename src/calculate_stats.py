import torch
import tqdm
from pathlib import Path
from dataset import TTSDataset


def calculate_stats(data_dir):
    """
    Calculates the global Mean and Std of Mel Spectrograms in the dataset.
    """
    data_dir = Path(data_dir)
    dataset = TTSDataset(str(data_dir))

    print(f"Scanning {len(dataset)} files in {data_dir}...")

    # We will use Welford's Online Algorithm or a simple two-pass approach.
    # Since the dataset fits in RAM (usually), we can try a batched approach,
    # but accumulating sum and sum_squared is safer for memory.

    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0

    # Pass 1: Accumulate sums
    # We iterate manually to show progress
    for i in tqdm.trange(len(dataset)):
        try:
            # item is a dict: {'mel_spectrograms': [Mel, Time], ...}
            item = dataset[i]
            mel = item['mel_spectrogram']

            # Ensure it's on CPU to save GPU memory
            mel = mel.to('cpu').float()

            total_sum += mel.sum().item()
            total_sq_sum += (mel ** 2).sum().item()
            total_elements += mel.numel()

        except Exception as e:
            print(f"Error loading file index {i}: {e}")
            continue

    # Calculate Global Mean and Std
    # mean = sum / N
    global_mean = total_sum / total_elements

    # variance = (sum_sq / N) - (mean^2)
    global_variance = (total_sq_sum / total_elements) - (global_mean ** 2)
    global_std = torch.sqrt(torch.tensor(global_variance)).item()

    print("\n" + "=" * 30)
    print("      DATASET STATISTICS      ")
    print("=" * 30)
    print(f"Total Elements: {total_elements}")
    print(f"Calculated Mean: {global_mean:.6f}")
    print(f"Calculated Std:  {global_std:.6f}")
    print("=" * 30)

    print("\nCopy these values into src/train_flow_matching.py AND src/inference.py:")
    print(f"MEL_MEAN = {global_mean:.6f}")
    print(f"MEL_STD  = {global_std:.6f}")


if __name__ == "__main__":
    # Point this to your processed data folder containing the .pt files
    calculate_stats("data/processed")