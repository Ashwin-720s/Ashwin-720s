"""
STEP 1 — Turn your CBSE textbook PDFs into clean, labeled text chunks.

WHAT THIS DOES:
1. Reads every PDF inside the `raw_pdfs/` folder.
2. Splits each PDF into small text chunks (a few sentences each).
3. Tries to guess the CHAPTER NAME for each chunk from the PDF filename
   (so make your filenames like: "Physics_Ch04_Thermodynamics.pdf").
4. Saves everything into `data/chunks.csv` — this file is the input
   for BOTH the classifier training (step 2) and the retrieval index (step 3).

HOW TO GET THE PDFs (do this before running the script):
- Go to https://ncert.nic.in/textbook.php (official NCERT site, free, legal).
- Pick Class 11 -> your subject (Physics / Chemistry / Biology / Maths).
- Download the chapter PDFs you want to cover (2-3 chapters per subject
  is PLENTY for a 1-week expo project — don't try to cover the whole book).
- Rename each file like: Subject_ChapterNumber_ChapterName.pdf
  e.g. "Physics_04_Thermodynamics.pdf", "Biology_02_BiologicalClassification.pdf"
- Put them all inside the raw_pdfs/ folder next to this script.

Run with:  python 1_prepare_data.py
"""

import os
import re
import csv
import fitz  # this is PyMuPDF

RAW_PDF_DIR = "raw_pdfs"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "chunks.csv")

# How many sentences roughly per chunk. Smaller = more precise retrieval,
# but too small loses context. 4-6 sentences is a good starting point.
SENTENCES_PER_CHUNK = 5


def clean_text(text):
    """Remove extra whitespace, page numbers, headers/footers noise."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+", "", text)
    return text.strip()


def split_into_sentences(text):
    # Simple sentence splitter — good enough for textbook prose.
    sentences = re.split(r"(?<=[.!?]) +", text)
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def chunk_sentences(sentences, size=SENTENCES_PER_CHUNK):
    for i in range(0, len(sentences), size):
        yield " ".join(sentences[i:i + size])


def parse_filename(filename):
    """
    Expects filenames like: Physics_04_Thermodynamics.pdf
    Returns (subject, chapter_label)
    Falls back gracefully if the filename doesn't follow the pattern.
    """
    name = os.path.splitext(filename)[0]
    parts = name.split("_")
    if len(parts) >= 3:
        subject = parts[0]
        chapter = " ".join(parts[1:]).replace("_", " ")
    elif len(parts) == 2:
        subject, chapter = parts[0], parts[1]
    else:
        subject, chapter = "Unknown", name
    return subject, chapter


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(RAW_PDF_DIR):
        os.makedirs(RAW_PDF_DIR, exist_ok=True)
        print(f"Created '{RAW_PDF_DIR}/' — put your CBSE textbook PDFs in there, then re-run this script.")
        return

    pdf_files = [f for f in os.listdir(RAW_PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in '{RAW_PDF_DIR}/'. Add some and re-run.")
        return

    rows = []
    chunk_id = 0

    for filename in pdf_files:
        subject, chapter = parse_filename(filename)
        path = os.path.join(RAW_PDF_DIR, filename)
        print(f"Reading {filename}  ->  subject='{subject}', chapter='{chapter}'")

        doc = fitz.open(path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        full_text = clean_text(full_text)
        sentences = split_into_sentences(full_text)

        for chunk in chunk_sentences(sentences):
            if len(chunk) < 40:
                continue  # skip near-empty chunks
            rows.append({
                "chunk_id": chunk_id,
                "subject": subject,
                "chapter": chapter,
                "text": chunk,
            })
            chunk_id += 1

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "subject", "chapter", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Wrote {len(rows)} text chunks to {OUTPUT_FILE}")
    print("Subjects found:", sorted(set(r["subject"] for r in rows)))
    print("Chapters found:", sorted(set(r["chapter"] for r in rows)))


if __name__ == "__main__":
    main()
