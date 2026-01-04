import os
import pickle
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_FILE = os.path.join("data", "pairs_filtered.csv")
INDEX_FILE = os.path.join("data", "faiss_index.bin")
META_FILE = os.path.join("data", "faiss_meta.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

def main():
    print("Loading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    queries = df["query"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    print("Loading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding queries...")
    embeddings = model.encode(
        queries,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dim = embeddings.shape[1]
    print("Embedding shape:", embeddings.shape)

    # FAISS index (cosine similarity via inner product because we normalized embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print("Saving index to:", INDEX_FILE)
    faiss.write_index(index, INDEX_FILE)

    print("Saving metadata to:", META_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"queries": queries, "answers": answers}, f)

    print("\nDONE ✅")
    print("Index size:", index.ntotal)

if __name__ == "__main__":
    main()
