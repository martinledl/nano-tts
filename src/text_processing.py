import torch
from g2p_en import G2p
from symbols import symbol_to_id

# Initialize G2P once (it loads the model)
# preserve_punctuation=True keeps '!', ',', '.' which we map to silence
g2p = G2p()


def text_to_sequence(text):
    """
    Converts raw text into a Tensor of IDs compatible with the trained model.
    Input: "Hello world!"
    Output: Tensor([4, 15, 88, ...])
    """

    # 1. G2P Conversion
    # g2p_en turns "Hello world!" into:
    # ['HH', 'AH0', 'L', 'OW1', ' ', 'W', 'ER1', 'L', 'D', '!', ' ', '!']
    phonemes = g2p(text)

    sequence = []

    for p in phonemes:
        # Remove whitespace (g2p_en produces ' ' tokens sometimes)
        if p == ' ':
            continue

        # Map Punctuation to "sil" (Silence)
        # This matches what MFA likely did in the TextGrids
        if p in [',', '.', '!', '?', ';', ':', '-']:
            # Only add silence if the model actually has a 'sil' or 'spn' token
            if 'sil' in symbol_to_id:
                sequence.append(symbol_to_id['sil'])
            elif 'spn' in symbol_to_id:
                sequence.append(symbol_to_id['spn'])
            continue

        # Map Phonemes
        if p in symbol_to_id:
            sequence.append(symbol_to_id[p])
        else:
            print(f"Warning: Unknown phoneme '{p}' - skipped.")

    return torch.LongTensor(sequence)


if __name__ == "__main__":
    # Test it
    print("Testing Inference G2P...")
    ids = text_to_sequence("Hello, World!")
    print(f"Generated IDs: {ids}")

    # Decode back to check
    from symbols import id_to_symbol

    decoded = [id_to_symbol[i.item()] for i in ids]
    print(f"Decoded: {decoded}")