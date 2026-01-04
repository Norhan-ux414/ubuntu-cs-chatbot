import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_FILE = os.path.join("data", "faiss_index.bin")
META_FILE = os.path.join("data", "faiss_meta.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

def load():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model

def search(query: str, index, meta, model, top_k=TOP_K):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        results.append({
            "score": float(score),
            "matched_query": meta["queries"][idx],
            "answer": meta["answers"][idx]
        })
    return results

def main():
    index, meta, model = load()
    print("✅ Retrieval test. Type your question (or 'exit'):\n")

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            break

        results = search(user_q, index, meta, model)

        print("\nTop matches:")
        for i, r in enumerate(results, 1):
            print(f"\n--- #{i}  (score={r['score']:.3f}) ---")
            print("Matched Q:", r["matched_query"])
            print("Answer   :", r["answer"])

        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
