import os
import csv
from pathlib import Path

LJ_HOME = Path("data/LJSpeech-1.1")
WAVS_DIR = LJ_HOME / "wavs"
METADATA_PATH = LJ_HOME / "metadata.csv"


def prepare_mfa_labels():
    if not METADATA_PATH.exists():
        print(f"Error: Could not find {METADATA_PATH}")
        return

    print(f"Generating .lab files in {WAVS_DIR}...")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        # LJSpeech uses | as delimiter. strict=True handles quoting issues.
        # Format: ID | Raw Text | Normalized Text
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)

        count = 0
        for row in reader:
            if len(row) < 3:
                continue

            file_id = row[0]
            # We use column 2 (Normalized Text) because it expands numbers
            # (e.g., "1990" -> "nineteen ninety"), which matches the audio better.
            text = row[2]

            # Write the text file
            lab_path = WAVS_DIR / f"{file_id}.lab"
            with open(lab_path, "w", encoding="utf-8") as lab_file:
                lab_file.write(text)

            count += 1

    print(f"Done! Created {count} .lab files.")


if __name__ == "__main__":
    prepare_mfa_labels()