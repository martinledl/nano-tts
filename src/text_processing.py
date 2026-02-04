import torch

# --- MONKEY PATCH START ---
# DeepPhonemizer uses an older pickle format that PyTorch 2.6 blocks by default.
# We intercept the torch.load call to force weights_only=False.

_original_load = torch.load


def unsafe_torch_load(*args, **kwargs):
    # Force weights_only=False if the caller didn't specify it
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)


# Apply the patch globally
torch.load = unsafe_torch_load

# Now import/run the library that was crashing
from dp.phonemizer import Phonemizer

try:
    phonemizer = Phonemizer.from_checkpoint("pretrained_models/en_us_cmudict_ipa_forward.pt")
except Exception as e:
    print("Error loading DeepPhonemizer model:", e)
    raise e
# --- MONKEY PATCH END ---

from g2p_en import G2p
import csv
from tqdm import tqdm
from symbols import MFA_SYMBOLS, IPA_SYMBOLS

# Initialize G2P model once
_g2p = G2p()


def text_to_sequence_g2p(text: str, cleaner_names: list[str] = None) -> list[int]:
    """
    Converts text to a sequence of phoneme IDs using g2p_en.
    Strictly filters for symbols present in MFA_SYMBOLS.
    """
    phonemes = _g2p(text)

    sequence = []

    for ph in phonemes:
        # We only keep the phoneme if it exists in your dictionary.
        # This automatically drops punctuation ('.', ',') and spaces (' ')
        # because they are not keys in MFA_SYMBOLS.
        if ph in MFA_SYMBOLS:
            sequence.append(MFA_SYMBOLS[ph])

    return sequence


def text_to_sequence(phonemizer: Phonemizer, text: str, lang: str = "en_us") -> list[int]:
    """
    Converts text to a sequence of phoneme IDs using DeepPhonemizer.
    """
    # 1. Get raw phoneme string (e.g. "həˈloʊ wɜrld")
    phoneme_string = phonemizer(text, lang=lang)

    # 2. Tokenize (Hard-coded to 'chars' for IPA checkpoint)
    # The IPA checkpoint outputs joined characters, so we iterate char by char.
    tokens = list(phoneme_string)

    sequence = []
    for ph in tokens:
        if ph == " ":
            # Map space to the 'spn' (silence) token if it exists
            if 'spn' in IPA_SYMBOLS:
                sequence.append(IPA_SYMBOLS['spn'])
            continue

        # 4. Handle Normal Symbols
        if ph in IPA_SYMBOLS:
            sequence.append(IPA_SYMBOLS[ph])
        else:
            print(f"Warning: Symbol '{ph}' not in dictionary, skipping.")

    print(f"Phonemized Text: '{phoneme_string}'")
    print(f"Phoneme Tokens: {tokens}")

    return sequence


if __name__ == "__main__":
    # Quick Test
    input_text = "Hey everyone, welcome to the TTS project!"
    seq_g2p = text_to_sequence_g2p(input_text)
    seq_dp = text_to_sequence(phonemizer, input_text)
    print("Input Text: ", input_text)
    print("G2P Sequence: ", seq_g2p)
    print("DeepPhonemizer Sequence: ", seq_dp)
