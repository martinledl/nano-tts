import re
from pathlib import Path
from tqdm import tqdm
from g2p_en import G2p


def build_dictionary(wavs_dir, output_dict_path):
    # Initialize the fast phonemizer
    g2p = G2p()
    wavs_path = Path(wavs_dir)

    print("1. Scanning text files for unique words...")
    words = set()
    lab_files = list(wavs_path.glob("*.lab"))

    for lab in tqdm(lab_files):
        with open(lab, "r", encoding="utf-8") as f:
            text = f.read().strip()
            # Clean punctuation for the dictionary keys
            # "Hello," -> "HELLO"
            clean_text = re.sub(r"[^\w\s']", '', text)

            for w in clean_text.split():
                # MFA dictionaries are usually UPPERCASE
                words.add(w.upper())

    print(f"Found {len(words)} unique words.")

    print("2. Generating pronunciations...")
    with open(output_dict_path, "w", encoding="utf-8") as f:
        # Add explicit Punctuation -> Silence mappings for MFA
        f.write(". sil\n")
        f.write(", sil\n")
        f.write("! sil\n")
        f.write("? sil\n")
        f.write("- sil\n")
        f.write(": sil\n")
        f.write("; sil\n")
        f.write("\" sil\n")

        for word in tqdm(words):
            if not word: continue

            # g2p(word) returns ['H', 'EH1', 'L', 'OW0']
            phonemes = g2p(word)

            # Filter out any lingering non-phoneme characters
            clean_phones = [p for p in phonemes if re.match(r'[A-Z]+[0-9]*', p)]

            if clean_phones:
                # Format: WORD PHONEME1 PHONEME2 ...
                entry = f"{word} {' '.join(clean_phones)}\n"
                f.write(entry)

    print(f"Success! Dictionary saved to {output_dict_path}")


if __name__ == "__main__":
    # Point this to your wavs folder
    build_dictionary("data/LJSpeech-1.1/wavs", "data/my_corpus_dict.txt")