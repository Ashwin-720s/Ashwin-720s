"""
STEP 2 (Unsloth version) — Turn textbook chunks into Q&A training data.

Fine-tuning needs QUESTION -> ANSWER pairs, not raw textbook paragraphs.
This script auto-generates those pairs using a local model (Ollama +
llama3.2:3b, running on your CPU) so you don't have to write hundreds of
Q&A pairs by hand. This step does NOT train anything — it just uses an
already-trained model to help you build a dataset. That's why it's fine
to run on CPU even though it's a bit slow (expect ~1-2 seconds per pair).

Requirements: Ollama installed and running, with llama3.2:3b pulled
  (see GUIDE.md section 2 for install steps)

Run with:  python 2_generate_qa_pairs.py
Input:     data/chunks.csv          (from 1_prepare_data.py)
Output:    data/training_data.json  (Alpaca format — instruction/input/output)
           This is the file you'll upload to Google Colab in Step 3.
"""

import os
import json
import re
import pandas as pd
import ollama

DATA_FILE = "data/chunks.csv"
OUTPUT_FILE = "data/training_data.json"
PAIRS_PER_CHUNK = 2  # how many Q&A pairs to generate per textbook chunk

PROMPT_TEMPLATE = """You are creating study Q&A pairs for a CBSE Class 11 {subject} textbook,
chapter "{chapter}". Based ONLY on the text below, write exactly {n} question-and-answer
pairs a student might ask while revising this chapter. Answers must be fully supported
by the text — do not add outside facts.

Text:
\"\"\"{text}\"\"\"

Respond with ONLY valid JSON, no other text, in this exact format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""


def extract_json_array(raw_text):
    """The model sometimes wraps JSON in markdown fences or adds commentary.
    This pulls out just the [...] array."""
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_pairs_for_chunk(subject, chapter, text):
    prompt = PROMPT_TEMPLATE.format(subject=subject, chapter=chapter, text=text, n=PAIRS_PER_CHUNK)
    try:
        response = ollama.generate(model="llama3.2:3b", prompt=prompt)
        pairs = extract_json_array(response["response"])
        return pairs or []
    except Exception as e:
        print(f"  (skipped a chunk — generation error: {e})")
        return []


def main():
    if not os.path.exists(DATA_FILE):
        print(f"Can't find {DATA_FILE}. Run 1_prepare_data.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Generating Q&A pairs from {len(df)} chunks (this will take a while — grab a snack)...")

    alpaca_data = []
    for i, row in df.iterrows():
        pairs = generate_pairs_for_chunk(row["subject"], row["chapter"], row["text"])
        for p in pairs:
            if "question" in p and "answer" in p:
                alpaca_data.append({
                    "instruction": p["question"],
                    "input": "",
                    "output": p["answer"],
                })
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(df)} chunks processed, {len(alpaca_data)} Q&A pairs so far")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Wrote {len(alpaca_data)} Q&A pairs to {OUTPUT_FILE}")
    print("This is the file you upload to the Colab notebook in Step 3 (see GUIDE.md).")

    if len(alpaca_data) < 50:
        print("\nHeads up: fewer than 50 examples is thin for fine-tuning.")
        print("Consider adding more chapters/PDFs and re-running 1_prepare_data.py + this script.")


if __name__ == "__main__":
    main()
