import csv
from pathlib import Path
from tqdm import tqdm
import re


def clean_text(text):
    # MFA works best with UPPERCASE for standard dictionaries
    text = text.upper()
    # Keep only letters, numbers, and basic punctuation
    text = re.sub(r"[^A-Z0-9\s.,?!'-]", "", text)
    return text


def prepare_text_for_mfa(metadata_path, wavs_dir):
    wavs_path = Path(wavs_dir)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        rows = list(reader)

    print("Generating clean .lab files (TEXT)...")

    for row in tqdm(rows):
        if len(row) < 3: continue
        file_id = row[0]
        # Use the 3rd column (Normalized Text) from LJSpeech
        text = row[2]

        cleaned = clean_text(text)

        # Write .lab file
        lab_path = wavs_path / f"{file_id}.lab"
        with open(lab_path, "w", encoding="utf-8") as lab_file:
            lab_file.write(cleaned)

    print("Done. .lab files now contain CLEAN TEXT (not phonemes).")


if __name__ == "__main__":
    prepare_text_for_mfa("data/LJSpeech-1.1/metadata.csv", "data/LJSpeech-1.1/wavs")