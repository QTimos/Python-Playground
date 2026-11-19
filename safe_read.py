from typing import Tuple
from pathlib import Path
import charset_normalizer


def safe_read(file_path: str, default_encoding: str = "utf-8", confidence: float = 0.9) -> Tuple[str, str]:
    raw = Path(file_path).read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode(encoding="utf-8-sig"), "utf-8-sig"

    try:
        return raw.decode(encoding="utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        pass

    best_match = charset_normalizer.from_bytes(raw).best()
    conf = best_match.coherence if best_match else 0.0
    enc = best_match.encoding if best_match else None
    if best_match is not None and enc is not None:
        if conf >= confidence:
            try:
                return raw.decode(encoding=enc, errors="strict"), enc
            except (UnicodeDecodeError, LookupError):
                pass

    try:
        return raw.decode(encoding=default_encoding, errors="strict"), default_encoding
    except UnicodeDecodeError:
        # Strict path
        raise UnicodeError(
                f"Unknown encoding for file '{file_path}', only got '{enc}' at"
                f" '{conf:.2f}' confidence."
                )
        # Lossy path
        # return raw.decode(encoding=default_encoding, errors="replace"), f"{default_encoding}-lossy"


def main() -> None:
    data, encoding = safe_read("RAG_theory.md")
    print(data)


if __name__ == "__main__":
    main()

