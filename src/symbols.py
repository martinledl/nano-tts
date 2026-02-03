MFA_SYMBOLS = {'HH': 0,
               'AE1': 1,
               'V': 2,
               'IH0': 3,
               'NG': 4,
               'F': 5,
               'AY1': 6,
               'ER0': 7,
               'D': 8,
               'DH': 9,
               'IH1': 10,
               'S': 11,
               'L': 12,
               'AA1': 13,
               'T': 14,
               '': 15,
               'TH': 16,
               'R': 17,
               'IY1': 18,
               'AH1': 19,
               'AH0': 20,
               'Z': 21,
               'M': 22,
               'UW1': 23,
               'AO1': 24,
               'K': 25,
               'G': 26,
               'B': 27,
               'W': 28,
               'AA0': 29,
               'N': 30,
               'P': 31,
               'EH1': 32,
               'IY0': 33,
               'EY2': 34,
               'AW1': 35,
               'SH': 36,
               'CH': 37,
               'OW1': 38,
               'UH1': 39,
               'ER1': 40,
               'UW2': 41,
               'ZH': 42,
               'AY2': 43,
               'EY1': 44,
               'JH': 45,
               'Y': 46,
               'spn': 47,
               'EH0': 48,
               'EH2': 49,
               'IH2': 50,
               'OW2': 51,
               'AO2': 52,
               'ER2': 53,
               'OW0': 54,
               'AA2': 55,
               'UW0': 56,
               'EY0': 57,
               'AE0': 58,
               'AE2': 59,
               'IY2': 60,
               'AW2': 61,
               'AH2': 62,
               'OY1': 63,
               'AO0': 64,
               'UH2': 65,
               'AY0': 66,
               'OY2': 67,
               'AW0': 68,
               'UH0': 69,
               'OY0': 70,
               '<pad>': 71
               }

MFA_SYMBOLS_INV = {v: k for k, v in MFA_SYMBOLS.items()}
PHONEME_LIST = [k for k in MFA_SYMBOLS.keys() if k != '<pad>']
VOCABULARY_SIZE = len(MFA_SYMBOLS)


def get_mfa_symbol_id(symbol: str) -> int:
    return MFA_SYMBOLS.get(symbol, MFA_SYMBOLS['<pad>'])


def get_mfa_symbol_from_id(symbol_id: int) -> str:
    return MFA_SYMBOLS_INV.get(symbol_id, '<pad>')


def get_phoneme_list() -> list[str]:
    return PHONEME_LIST


def get_vocabulary_size() -> int:
    return VOCABULARY_SIZE


def get_padding_symbol_id() -> int:
    return MFA_SYMBOLS['<pad>']


