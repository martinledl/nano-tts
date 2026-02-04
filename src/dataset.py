import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from symbols import symbol_to_id


class TTSDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = list(Path(data_dir).glob("*.pt"))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx], weights_only=True)
        return data


class TTSCollate:
    def __call__(self, batch):
        # Sort batch by input lengths (descending)
        batch = sorted(batch, key=lambda x: x['phonemes'].shape[0], reverse=True)

        phonemes = [item['phonemes'] for item in batch]
        durations = [item['durations'] for item in batch]
        mel_spectrograms = [item['mel_spectrogram'] for item in batch]

        phoneme_lengths = torch.tensor([p.shape[0] for p in phonemes], dtype=torch.long)
        mel_lengths = torch.tensor([m.shape[-1] for m in mel_spectrograms], dtype=torch.long)

        padded_phonemes = pad_sequence(phonemes, batch_first=True, padding_value=symbol_to_id["pad"])
        padded_durations = pad_sequence(durations, batch_first=True, padding_value=0)

        # Transpose to (Time, 80) so pad_sequence treats Time as the length
        mels_transposed = [m.transpose(0, 1) for m in mel_spectrograms]
        padded_mels_transposed = pad_sequence(mels_transposed, batch_first=True, padding_value=0.0)
        # Transpose back to (Batch, 80, Max_Time) for the model
        padded_mel_spectrograms = padded_mels_transposed.transpose(1, 2)

        return {
            'phonemes': padded_phonemes,
            'durations': padded_durations,
            'mel_spectrograms': padded_mel_spectrograms,
            'phoneme_lengths': phoneme_lengths,
            'mel_lengths': mel_lengths
        }


if __name__ == "__main__":
    # Setup
    dataset = TTSDataset("data/processed")
    collate_fn = TTSCollate()
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    # Get one batch
    batch = next(iter(loader))

    print("Phonemes:", batch["phonemes"].shape)
    print("Durations:", batch["durations"].shape)
    print("Mels:", batch["mel_spectrograms"].shape)
    print("Text Lengths:", batch["phoneme_lengths"])
    print("Mel Lengths:", batch["mel_lengths"])

    # Validation
    assert batch["phonemes"].shape[0] == 16
    assert batch["mel_spectrograms"].shape[1] == 80  # Should be [Batch, 80, Time]