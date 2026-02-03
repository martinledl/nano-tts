from g2p_en import G2p
from symbols import MFA_SYMBOLS

# Initialize G2P model once
_g2p = G2p()

def text_to_sequence(text: str, cleaner_names: list[str] = None) -> list[int]:
    """
    Converts text to a sequence of phoneme IDs using g2p_en.
    Strictly filters for symbols present in MFA_SYMBOLS.
    """
    # 1. Convert Text to Phonemes
    # Output example: ['HH', 'AH0', 'L', 'OW1', ' ', 'W', 'ER1', 'L', 'D', '.', ' ', ' ']
    phonemes = _g2p(text)
    
    sequence = []
    
    for ph in phonemes:
        # 2. Strict Filtering
        # We only keep the phoneme if it exists in your dictionary.
        # This automatically drops punctuation ('.', ',') and spaces (' ')
        # because they are not keys in MFA_SYMBOLS.
        if ph in MFA_SYMBOLS:
            sequence.append(MFA_SYMBOLS[ph])
            
    return sequence

if __name__ == "__main__":
    # Quick Test
    input_text = "Hello world!"
    ids = text_to_sequence(input_text)
    print(f"Input: '{input_text}'")
    print(f"Phoneme IDs: {ids}")