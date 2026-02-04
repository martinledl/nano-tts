import torch
import csv
from tqdm import tqdm
from pathlib import Path

# --- MONKEY PATCH (Required for DeepPhonemizer + PyTorch 2.6+) ---
_original_load = torch.load


def unsafe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


torch.load = unsafe_torch_load

from dp.phonemizer import Phonemizer


# --- END PATCH ---


def extract_and_save_symbols(metadata_path, phonemizer_path,):
    # 1. Load Phonemizer
    print(f"Loading Phonemizer from {phonemizer_path}...")
    phonemizer = Phonemizer.from_checkpoint(phonemizer_path)

    # 2. Setup Set for Unique Symbols
    unique_symbols = set()

    # Pre-add special tokens we definitely need
    unique_symbols.add("pad")  # Padding
    unique_symbols.add("spn")  # Silence/Space (MFA style) or just space
    unique_symbols.add("eos")  # End of sequence (optional but good)

    print(f"Reading metadata from {metadata_path}...")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        rows = list(reader)

    print("Extracting symbols... (This might take a few minutes)")

    # We scan a subset first to check the format, then run all
    for i, row in enumerate(tqdm(rows)):
        if len(row) < 3: continue

        text = row[2]  # Normalized text

        # Run inference
        # Note: lang='en_us' is standard for that checkpoint
        phoneme_string = phonemizer(text, lang='en_us')

        # --- FORMAT CHECK (Run once on first item) ---
        if i == 0:
            print("\n" + "=" * 40)
            print(f"DEBUG: Sample Output: '{phoneme_string}'")
            if " " in phoneme_string and len(phoneme_string.split()) > len(text.split()):
                print("DETECTED: Space-separated phonemes. Using split().")
                mode = "split"
            else:
                print("DETECTED: Joined phonemes (or char-level). Using list() to get characters.")
                mode = "chars"
            print("=" * 40 + "\n")
        # ---------------------------------------------

        if mode == "split":
            # If phonemes are "h ə l l o", we split by space
            tokens = phoneme_string.strip().split()
        else:
            # If phonemes are "həllo", we treat every character as a symbol
            # We also filter out empty spaces if they are just word boundaries
            tokens = list(phoneme_string)

        # Add to set
        for token in tokens:
            if token.strip():  # Ignore pure whitespace if it wasn't caught
                unique_symbols.add(token)

    # 3. Sort and Format
    # We sort to ensure deterministic ID assignment
    sorted_symbols = sorted(list(unique_symbols))

    # 4. Print Resulting Dict
    print("\n# Generated Symbols Dictionary")
    print("IPA_SYMBOLS = {")
    for i, symbol in enumerate(sorted_symbols):
        print(f"    '{symbol}': {i},")
    print("}")

    print(f"\nSuccess! Extracted {len(sorted_symbols)} unique symbols.")
    print("Replace your old symbols dict with this.")


if __name__ == "__main__":
    # Update paths as needed
    METADATA = "data/LJSpeech-1.1/metadata.csv"
    PHONEMIZER_CKPT = "pretrained_models/en_us_cmudict_ipa_forward.pt"

    extract_and_save_symbols(METADATA, PHONEMIZER_CKPT)