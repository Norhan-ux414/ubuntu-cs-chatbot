import os
import re
import pandas as pd

INPUT_FILE = os.path.join("data", "pairs_clean.csv")
OUTPUT_FILE = os.path.join("data", "pairs_filtered.csv")

MIN_Q_LEN = 15
MIN_A_LEN = 15
MAX_LEN = 400  # نشيل اللي طويل اوي لأنه غالباً spam/chat

# كلمات تشير للدعم الفني (Ubuntu)
TECH_HINTS = [
    "ubuntu", "apt", "dpkg", "sudo", "bash", "terminal", "kernel", "grub",
    "wifi", "network", "drivers", "nvidia", "bluetooth", "update", "upgrade",
    "install", "package", "error", "failed", "permission", "mount", "disk"
]

def normalize(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def has_tech_signal(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in TECH_HINTS) or ("sudo " in t) or ("apt " in t) or ("error" in t)

def looks_like_noise(text: str) -> bool:
    t = text.lower()
    # روابط كتير أو ايميلات
    if t.count("http") >= 1:
        return True
    # جملة كلها رموز
    if sum(c.isalnum() for c in t) / max(len(t), 1) < 0.55:
        return True
    return False

def main():
    df = pd.read_csv(INPUT_FILE)

    # normalize
    df["query"] = df["query"].map(normalize)
    df["answer"] = df["answer"].map(normalize)

    # length filters
    df = df[df["query"].str.len().between(MIN_Q_LEN, MAX_LEN)]
    df = df[df["answer"].str.len().between(MIN_A_LEN, MAX_LEN)]

    # remove noisy rows
    df = df[~df["query"].map(looks_like_noise)]
    df = df[~df["answer"].map(looks_like_noise)]

    # keep only rows with tech signal in query OR answer
    df = df[df.apply(lambda r: has_tech_signal(r["query"]) or has_tech_signal(r["answer"]), axis=1)]

    df = df.drop_duplicates()

    print("Filtered shape:", df.shape)
    print("Saving:", OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSamples:")
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
