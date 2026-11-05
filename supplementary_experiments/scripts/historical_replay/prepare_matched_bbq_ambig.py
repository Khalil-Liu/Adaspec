"""Build the ambiguous BBQ test subset paired with the archived disambiguated test set.

The official category JSONL files use adjacent example IDs for the ambiguous
and disambiguated versions of each question. Selection is based on the already
fixed disambiguated test split, not on answers or model outcomes.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "BBQ-main" / "BBQ-main" / "data"
DISAMBIG = ROOT / "sea-llm" / "data" / "BBQ" / "disambig" / "All-test-disambig.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "bbq_ambig_input"
OUTPUT = OUTPUT_DIR / "All-test-ambig.jsonl"
EXPECTED_DISAMBIG_SHA256 = "2fb0942d7a1f5d7fe9b7c83b1c89e611c629812987e86c011f62d2952913f6e9"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    digest = sha256(DISAMBIG)
    if digest != EXPECTED_DISAMBIG_SHA256:
        raise ValueError(f"Disambiguated test SHA256 changed: {digest}")
    test = read_jsonl(DISAMBIG)
    if len(test) != 5246:
        raise ValueError(f"Unexpected disambiguated test size: {len(test)}")

    files = sorted(SOURCE.glob("*.jsonl"))
    if len(files) != 11:
        raise ValueError(f"Expected 11 BBQ category files, found {len(files)}")
    ambiguous = {}
    source_hashes = {}
    for path in files:
        source_hashes[path.name] = sha256(path)
        for row in read_jsonl(path):
            if row["context_condition"] != "ambig":
                continue
            key = (row["category"], row["example_id"])
            if key in ambiguous:
                raise ValueError(f"Duplicate ambiguous key: {key}")
            ambiguous[key] = row

    paired = []
    for row in test:
        key = (row["category"], row["example_id"] - 1)
        match = ambiguous.get(key)
        if match is None:
            raise ValueError(f"Missing ambiguous pair: {key}")
        if match["question"] != row["question"] or match["question_polarity"] != row["question_polarity"]:
            raise ValueError(f"Question or polarity mismatch: {key}")
        paired.append(match)
    if len({(r["category"], r["example_id"]) for r in paired}) != len(paired):
        raise ValueError("Duplicate selected ambiguous row")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        for row in paired:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "selection": "Ambiguous pair of each row in the fixed disambiguated test split, by adjacent official example ID; no label- or prediction-based selection",
        "n": len(paired),
        "source_dir": str(SOURCE),
        "source_sha256": source_hashes,
        "disambig_test_sha256": digest,
        "ambig_test_sha256": sha256(OUTPUT),
        "checks": "5246 unique pairs; category, question, and question polarity match for every pair",
    }
    (OUTPUT_DIR / "All-test-ambig.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(OUTPUT), "n": len(paired), "sha256": manifest["ambig_test_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
