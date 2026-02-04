import textgrid
from pathlib import Path
from tqdm import tqdm


def extract_symbols(aligned_dir, output_path):
    aligned_path = Path(aligned_dir)
    files = list(aligned_path.glob("*.TextGrid"))

    unique_symbols = set()

    print(f"Scanning {len(files)} TextGrids...")
    for p in tqdm(files):
        try:
            tg = textgrid.TextGrid.fromFile(str(p))
            # MFA usually puts phones in the 'phones' tier
            tier = tg.getFirst("phones")
            for interval in tier.intervals:
                sym = interval.mark
                # MFA uses "" or "<sil>" for silence. We track them but don't add to list yet.
                if sym not in ["", "<sil>", "spn", "sil"]:
                    unique_symbols.add(sym)
        except Exception:
            pass

    # Sort and Create List
    sorted_phones = sorted(list(unique_symbols))

    # Define Specials
    specials = ["pad", "spn", "eos", "sil"]

    final_symbols = specials + sorted_phones

    print(f"Found {len(sorted_phones)} unique phones.")
    print(f"Total symbols (with specials): {len(final_symbols)}")

    content = f"""# Auto-extracted from MFA TextGrids
symbols = {final_symbols}
symbol_to_id = {{s: i for i, s in enumerate(symbols)}}
id_to_symbol = {{i: s for i, s in enumerate(symbols)}}
"""

    with open(output_path, "w") as f:
        f.write(content)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    extract_symbols("data/aligned", "src/symbols.py")