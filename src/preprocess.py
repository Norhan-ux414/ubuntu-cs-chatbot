import os
import re
import pandas as pd
from tqdm import tqdm

INPUT_FILE = os.path.join("data", "dialogueText.csv")   
OUTPUT_FILE = os.path.join("data", "pairs_clean.csv")

MIN_LEN_Q = 8
MIN_LEN_A = 8

BAD_SHORT = {"ok", "okay", "k", "thx", "thanks", "ty", "lol", "yes", "no", "yep", "nope"}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # remove obvious IRC artifacts
    s = re.sub(r"^\[.*?\]\s*", "", s)   
    return s.strip()

def is_bad_utterance(s: str) -> bool:
    if not s:
        return True
    low = s.lower().strip()
    if low in BAD_SHORT:
        return True
    if len(low) < MIN_LEN_Q:
        return True
    # too many symbols
    if sum(c.isalnum() for c in low) / max(len(low), 1) < 0.5:
        return True
    return False

def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(clean_text)
    df = df.dropna(subset=["dialogueID", "from", "text"])
    df = df.sort_values(["dialogueID", "date"])

    pairs = []
    # group per dialogue
    for did, g in tqdm(df.groupby("dialogueID"), total=df["dialogueID"].nunique()):
        msgs = g[["from", "text"]].values
        for i in range(len(msgs) - 1):
            speaker1, q = msgs[i]
            speaker2, a = msgs[i + 1]

            if speaker1 == speaker2:
                continue  # same speaker, skip

            if is_bad_utterance(q) or is_bad_utterance(a):
                continue

            pairs.append((q, a))

    out = pd.DataFrame(pairs, columns=["query", "answer"])
    return out

def main():
    print("Reading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    print("Building pairs...")
    pairs_df = build_pairs(df)

    print("Pairs shape:", pairs_df.shape)
    print("Saving to:", OUTPUT_FILE)
    pairs_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSample pairs:")
    print(pairs_df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
