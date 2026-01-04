import os
import re
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# =========================
# Files
# =========================
INDEX_FILE = os.path.join("data", "faiss_index.bin")
META_FILE = os.path.join("data", "faiss_meta.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 20

# =========================
# Intent Filter
# =========================
UBUNTU_INTENT_HINTS = [
    "ubuntu", "linux", "apt", "apt-get", "dpkg", "sudo", "kernel", "grub",
    "wifi", "wireless", "network", "nmcli", "bluetooth", "nvidia", "driver",
    "install", "update", "upgrade", "error", "failed", "package", "terminal",
    "bash", "permission", "mount", "disk", "snap", "repo", "dependency"
]

def is_ubuntu_question(q: str) -> bool:
    q = q.lower().strip()
    return any(h in q for h in UBUNTU_INTENT_HINTS)

# =========================
# Load assets (cached)
# =========================
@st.cache_resource
def load_assets():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model

def search(query: str, index, meta, model, top_k=TOP_K):
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
# Reranking
# =========================
BAD_PATTERNS = [
    r"#\w+",
    r"\bjoin\b",
    r"\blol\b",
    r"\bwrong channel\b",
    r"\bask on\b",
    r"\birssi\b",
]

GOOD_HINTS = [
    "sudo", "apt", "apt-get", "dpkg", "systemctl", "service", "nmcli",
    "ifconfig", "iwconfig", "lspci", "lsusb", "dmesg",
    "reboot", "restart", "update", "upgrade", "install",
    "kernel", "driver", "dependency", "failed", "error"
]

def answer_quality(ans: str) -> float:
    a = (ans or "").lower().strip()

    for pat in BAD_PATTERNS:
        if re.search(pat, a):
            return -2.0

    score = 0.0
    for g in GOOD_HINTS:
        if g in a:
            score += 0.25

    L = len(a)
    if 30 <= L <= 350:
        score += 0.5
    elif L < 15:
        score -= 0.5
    elif L > 600:
        score -= 0.3

    if any(x in a for x in ["try", "run", "check", "type", "command"]):
        score += 0.3

    return score

def pick_best(results):
    best = None
    best_total = -999.0

    for r in results:
        r["qual"] = answer_quality(r["answer"])
        r["total"] = r["sim"] + r["qual"]
        if r["total"] > best_total:
            best_total = r["total"]
            best = r

    ranked = sorted(results, key=lambda x: x["total"], reverse=True)
    return best, ranked

# =========================
# Command extraction (clean)
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
            p = re.sub(r"^\s*run\s+", "", p, flags=re.IGNORECASE)

            if p.lower().startswith(("apt-get", "apt ")):
                p = "sudo " + p

            if len(p) >= 5:
                commands.append(p)

    if "reboot" in t.lower():
        commands.append("sudo reboot")

    seen = set()
    unique = []
    for c in commands:
        if c not in seen:
            unique.append(c)
            seen.add(c)

    return unique[:8]

def format_support_answer(best_answer: str) -> str:
    cmds = extract_commands(best_answer)

    out = []
    out.append("✅ **Suggested fix (based on similar Ubuntu issues):**\n")
    out.append("**1) Best next step:**")
    out.append(f"- {best_answer.strip()}")

    if cmds:
        out.append("\n**2) Commands to run (copy/paste):**")
        out.append("```bash")
        for c in cmds:
            out.append(c)
        out.append("```")

    out.append("\n**3) If it still doesn't work, reply with:**")
    out.append("- Ubuntu version: `lsb_release -a`")
    out.append("- WiFi chipset: `lspci | grep -i net`")
    out.append("- Network status: `nmcli dev status`")
    out.append("- Logs: `dmesg | tail -50`")

    return "\n".join(out)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Ubuntu Customer Service Chatbot", page_icon="🐧")
st.title("🐧 Ubuntu Automated Customer Service Chatbot")
st.write("Retrieval-based chatbot using **FAISS + Sentence Transformers + reranking**")

index, meta, model = load_assets()

# session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns([4, 1])
with col1:
    user_q = st.text_input("Ask an Ubuntu support question:", key="user_input")
with col2:
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if st.button("Send"):
    if user_q.strip():
        # store user message
        st.session_state.messages.append(("user", user_q.strip()))

        # assistant response
        if not is_ubuntu_question(user_q):
            bot_msg = (
                "🤖 I can only help with **Ubuntu / Linux technical support** questions.\n\n"
                "Try: `wifi not working after update` or `apt-get update failed`."
            )
        else:
            results = search(user_q, index, meta, model)
            best, ranked = pick_best(results)
            bot_msg = format_support_answer(best["answer"])

        st.session_state.messages.append(("bot", bot_msg))
        st.rerun()


# display chat
st.divider()
for role, msg in st.session_state.messages:
    if role == "user":
        st.markdown(f"🧑 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:**\n\n{msg}")