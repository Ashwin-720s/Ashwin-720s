"""
STEP 3 — Build the retrieval index. This is what lets the bot find the
right piece of textbook text to answer a question, instead of guessing.

No training happens here — we use a pretrained embedding model
(all-MiniLM-L6-v2, ~80MB, runs fast on CPU) to convert every text chunk
into a vector, then store those vectors in a FAISS index for fast search.

Run with:  python 3_build_retrieval_index.py
Outputs:
  models/faiss_index.bin   <- the searchable vector index
  models/chunk_lookup.joblib <- maps index positions back to original text
"""

import os
import pandas as pd
import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/chunks.csv"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"Can't find {DATA_FILE}. Run 1_prepare_data.py first.")
        return

    df = pd.read_csv(DATA_FILE)
    print(f"Embedding {len(df)} chunks... (this may take a couple of minutes on CPU)")

    # Small, fast, CPU-friendly embedding model.
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        df["text"].tolist(),
        show_progress_bar=True,
        batch_size=32,
    )
    embeddings = np.array(embeddings).astype("float32")

    # Normalize so we can use inner product as cosine similarity.
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product (cosine sim after normalizing)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
    joblib.dump(df, os.path.join(MODEL_DIR, "chunk_lookup.joblib"))

    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dimension}) to {MODEL_DIR}/faiss_index.bin")


if __name__ == "__main__":
    main()
