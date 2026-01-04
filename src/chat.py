import os
import re
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# Files
# =========================
INDEX_FILE = os.path.join("data", "faiss_index.bin")
META_FILE = os.path.join("data", "faiss_meta.pkl")

# =========================
# Model
# =========================
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# Retrieval / Debug
# =========================
TOP_K_RETRIEVE = 20
SHOW_DEBUG = False

MIN_TOTAL_SCORE = 1.10

# =========================
# Rerank rules
# =========================
BAD_PATTERNS = [
    r"#\w+",          # #ubuntu+1
    r"\bjoin\b",
    r"\birssi\b",
    r"\blol\b",
    r"\bpm me\b",
    r"\bgoogle it\b",
    r"\bask on\b",
    r"\bwrong channel\b",
    r"\bask in\b",
    r"\bgo to #\b",
]

GOOD_HINTS = [
    "sudo", "apt", "apt-get", "dpkg", "systemctl", "service", "nmcli",
    "ifconfig", "iwconfig", "lspci", "lsusb", "modprobe", "dmesg",
    "/etc/", "reboot", "restart", "update", "upgrade", "install",
    "purge", "remove", "networkmanager", "netplan", "rfkill",
    "error", "failed", "dependency", "dependencies", "kernel", "driver"
]
UBUNTU_INTENT_HINTS = [
    "ubuntu", "linux", "apt", "apt-get", "dpkg", "sudo", "kernel", "grub",
    "wifi", "wireless", "network", "nmcli", "bluetooth", "nvidia", "driver",
    "install", "update", "upgrade", "error", "failed", "package", "terminal",
    "bash", "permission", "mount", "disk"
]

def is_ubuntu_question(q: str) -> bool:
    q = q.lower().strip()
    return any(h in q for h in UBUNTU_INTENT_HINTS)


# =========================
# Load Index + Meta + Model
# =========================
def load():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"Missing index file: {INDEX_FILE}")
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Missing metadata file: {META_FILE}")

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model


# =========================
# Answer quality scoring (rerank)
# =========================
def answer_quality_score(ans: str) -> float:
    a = (ans or "").lower().strip()

    # hard penalty for chat/noise patterns
    for pat in BAD_PATTERNS:
        if re.search(pat, a):
            return -2.0

    score = 0.0

    # reward technical hints
    for w in GOOD_HINTS:
        if w in a:
            score += 0.25

    # reward reasonable length
    L = len(a)
    if 30 <= L <= 350:
        score += 0.5
    elif L < 15:
        score -= 0.5
    elif L > 600:
        score -= 0.3

    # reward step language
    if any(x in a for x in ["try", "run", "check", "edit", "open", "type", "command", "reboot", "restart"]):
        score += 0.3

    return score


# =========================
# Retrieval search
# =========================
def search(query: str, index, meta, model, top_k=TOP_K_RETRIEVE):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(q_emb, top_k)

    results = []
    for sim, idx in zip(scores[0], ids[0]):
        results.append({
            "sim": float(sim),
            "matched_query": meta["queries"][idx],
            "answer": meta["answers"][idx],
        })
    return results


# =========================
# Pick best by reranking
# =========================
def pick_best(results):
    best = None
    best_score = -999.0

    for r in results:
        qual = answer_quality_score(r["answer"])
        total = r["sim"] + qual
        r["qual"] = qual
        r["total"] = total

        if total > best_score:
            best_score = total
            best = r

    ranked = sorted(results, key=lambda x: x["total"], reverse=True)
    return best, ranked


# =========================
# Extract commands (clean)
# =========================
def extract_commands(text: str):
    if not text:
        return []

    t = text.strip()
    parts = re.split(r",|&&|\.\s+", t)

    cmd_keywords = [
        "sudo", "apt-get", "apt ", "nmcli", "ifconfig", "iwconfig",
        "lspci", "lsusb", "dmesg", "systemctl", "rfkill", "netplan"
    ]

    commands = []

    for p in parts:
        p = p.strip()
        if not p:
            continue

        low = p.lower()

        if any(k in low for k in cmd_keywords):
            #  remove leading "run "
            p = re.sub(r"^\s*run\s+", "", p, flags=re.IGNORECASE)

            #  add sudo if apt/apt-get without sudo
            if p.lower().startswith(("apt-get", "apt ")):
                p = "sudo " + p

            if len(p) >= 5:
                commands.append(p)

    #  If the answer mentions reboot, add reboot command
    if "reboot" in t.lower():
        commands.append("sudo reboot")

    # remove duplicates preserving order
    seen = set()
    unique = []
    for c in commands:
        if c not in seen:
            unique.append(c)
            seen.add(c)

    return unique[:8]


# =========================
# Format final customer service response
# =========================
def format_support_answer(user_q: str, best_answer: str) -> str:
    commands = extract_commands(best_answer)

    response = []
    response.append(" Suggested fix (based on similar Ubuntu issues):\n")

    response.append("1) Best next step:")
    response.append(f"- {best_answer.strip()}")

    if commands:
        response.append("\n2) Commands to run (copy/paste):")
        response.append("```bash")
        for c in commands:
            response.append(c)
        response.append("```")

    response.append("\n3) If it still doesn't work, reply with:")
    response.append("- Ubuntu version: `lsb_release -a`")
    response.append("- WiFi chipset: `lspci | grep -i net`")
    response.append("- Network status: `nmcli dev status`")
    response.append("- Logs: `dmesg | tail -50`")

    return "\n".join(response)


# =========================
# Main loop
# =========================
def main():
    print("Starting chatbot...\n")
    index, meta, model = load()

    print(" Chatbot (retrieval + rerank). Type your question (or 'exit'):\n")

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            break
        if not user_q:
            continue
        if not is_ubuntu_question(user_q):
         print("\n🤖 I can only help with Ubuntu / Linux technical support questions.")
         print("Try asking something like: 'wifi not working after update' or 'apt-get update failed'.")
         print("\n" + "=" * 60 + "\n")
         continue

        results = search(user_q, index, meta, model)
        best, ranked = pick_best(results)

        #  threshold: avoid hallucinated answers for non-Ubuntu questions
        if best["total"] < MIN_TOTAL_SCORE:
            print("\n🤖 I’m not confident I have a good match for this question.")
            print("If this is an Ubuntu issue, please reply with:")
            print("- Ubuntu version: lsb_release -a")
            print("- Hardware: lspci | grep -i net")
            print("- Logs: dmesg | tail -50")
            print("\n" + "=" * 60 + "\n")
            continue

        print("\n🤖 Best answer:\n")
        print(format_support_answer(user_q, best["answer"]))

        if SHOW_DEBUG:
            print("\n--- Debug top 5 (total = sim + quality) ---")
            for i, r in enumerate(ranked[:5], 1):
                print(f"\n#{i} total={r['total']:.3f}  sim={r['sim']:.3f}  qual={r['qual']:.3f}")
                print("Matched Q:", r["matched_query"])
                print("Answer   :", r["answer"])

        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
