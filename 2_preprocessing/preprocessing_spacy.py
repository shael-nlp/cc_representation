import spacy
from spacy.tokens import DocBin
from pathlib import Path
import os

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("processed_docs")
MODEL = "en_core_web_lg"

nlp = spacy.load(MODEL)

print(f"Processing .txt files from: {INPUT_DIR}")
text_files = list(INPUT_DIR.glob("*.txt"))

for text_file_path in sorted(text_files):
    print(f"Processing: {text_file_path.name}...")

    with open(text_file_path, "r", encoding="utf-8") as f:
        text_content = f.read()

    doc = nlp(text_content)
    doc_bin = DocBin(docs=[doc])
    output_file_path = OUTPUT_DIR / (text_file_path.stem + ".spacy")

    doc_bin.to_disk(output_file_path)
    print(f"Successfully saved: {output_file_path}")

print("Preprocessing complete.")