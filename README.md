#  Ubuntu Automated Customer Service Chatbot

A retrieval-based chatbot for Ubuntu technical support using the Ubuntu Dialogue Corpus.
It uses Sentence Transformers embeddings + FAISS vector search + a lightweight reranker to return helpful support-style answers.

---

##  Features
- Retrieval-based Q/A chatbot (no hallucination from generation)
- Sentence Transformers embeddings (`all-MiniLM-L6-v2`)
- FAISS similarity search for fast retrieval
- Reranking based on technical answer quality
- Intent filter: rejects non-Ubuntu / non-technical questions
- Streamlit chat UI with conversation history

---

##  Project Structure

ubuntu-cs-chatbot/
│── app.py # Streamlit UI
│── data/
│ ├── pairs_filtered.csv # filtered Q/A pairs
│ ├── faiss_index.bin # FAISS index
│ ├── faiss_meta.pkl # metadata (queries + answers)
│── src/
│ ├── inspect_data.py
│ ├── preprocess.py
│ ├── filter_pairs.py
│ ├── build_index.py
│ ├── chat.py # terminal chatbot version
│── notebooks/
│── .venv/
│── README.md


---

##  Setup

### 1) Create and activate virtual environment (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

###2) Install dependencies
pip install -U pip
pip install pandas scikit-learn faiss-cpu sentence-transformers streamlit

#Dataset

Ubuntu Dialogue Corpus (Kaggle)

The dataset contains multi-turn Ubuntu IRC support conversations.
We convert it into (query, answer) pairs and remove noisy or irrelevant pairs.

##Pipeline
1) Inspect dataset
python src/inspect_data.py

2) Preprocess and build Q/A pairs
python src/preprocess.py

3) Filter noisy pairs
python src/filter_pairs.py

4) Build FAISS index (embeddings)
python src/build_index.py

##Run the chatbot
- Terminal version
python src/chat.py

- Streamlit UI version
streamlit run app.py

##Evaluation 

We evaluate the chatbot by:

Manual testing on common Ubuntu issues (WiFi, apt, drivers, boot errors)

Checking if retrieved answers contain relevant troubleshooting steps and commands

Ensuring non-Ubuntu questions are rejected by the intent filter

- Future Improvements

Train a dedicated reranker model (cross-encoder) for better ranking

Add multi-turn context handling

Add automatic issue categorization (WiFi, apt, drivers, etc.)

Deploy to cloud (Streamlit Cloud / HuggingFace Spaces)