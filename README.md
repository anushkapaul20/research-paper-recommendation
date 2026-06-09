# 📄 Research Paper Recommendation System

A semantic search engine that recommends research papers similar to yours. Upload a PDF or enter a text query — the system finds the most relevant papers from a live ArXiv dataset using sentence embeddings and FAISS vector search.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![License](https://img.shields.io/badge/License-MIT-green)

🔗 **Live Demo:** [research-paper-recommendation.streamlit.app](https://research-paper-recommendation.streamlit.app)

---

## 🚀 Demo

| Upload PDF | Text Query | Similar Papers |
|---|---|---|
| Upload any research paper PDF | Type a title or abstract | Get ranked similar papers with scores |

---

## 🧠 How It Works

```
PDF / Text Query
      │
      ▼
Text Extraction (PyMuPDF)
      │
      ▼
Abstract Preprocessing & Cleaning
      │
      ▼
Semantic Embedding (all-MiniLM-L6-v2, 384-dim)
      │
      ▼
FAISS Cosine Similarity Search (1,500+ ArXiv papers)
      │
      ▼
Ranked Recommendations + Authors + Research Directions
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (IndexFlatIP) |
| Dataset | ArXiv API (~1,500 papers across 10 domains) |
| PDF Parsing | PyMuPDF (fitz) |
| ML Framework | PyTorch |

---

## 📁 Project Structure

```
research-paper-recommendation/
├── app.py                  # Streamlit UI (main entry point)
├── requirements.txt        # Python dependencies
├── README.md
├── data/
│   └── papers.csv          # Auto-generated after first fetch
├── models/
│   ├── faiss.index         # Auto-generated FAISS index
│   └── metadata.pkl        # Paper metadata store
└── src/
    ├── __init__.py
    ├── data_collector.py   # ArXiv API paper fetching
    ├── text_processor.py   # PDF extraction & text cleaning
    ├── embedder.py         # Sentence Transformer embeddings
    ├── vector_store.py     # FAISS index management
    └── recommender.py      # Pipeline orchestration
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/research-paper-recommendation.git
cd research-paper-recommendation
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** On some systems, `tokenizers` may need Rust to build from source.  
> If you get a build error, run: `pip install tokenizers --prefer-binary`

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🖥️ First-Time Use

1. Open `http://localhost:8501` in your browser
2. In the **sidebar**, click **"Fetch Papers & Build Index"**
   - Downloads ~1,500 papers from ArXiv across 10 ML/CS domains
   - Builds the FAISS semantic search index
   - Takes ~5 minutes on first run
3. Use **Upload PDF** tab to upload your research paper, or
4. Use **Text Query** tab to type a title/abstract

---

## ✨ Features

- **PDF Upload** — extracts abstract automatically from any research paper PDF
- **Text Query** — search by typing a title, abstract, or research description
- **Semantic Search** — finds papers by meaning, not just keyword matching
- **Similarity Scores** — each result shows a cosine similarity percentage
- **Author Recommendations** — suggests frequent authors in the related area
- **Research Directions** — infers suggested next research directions
- **Keyword Extraction** — pulls key terms from your query
- **CSV Export** — download your recommendations as a spreadsheet
- **Dataset Management** — fetch and rebuild the index from the sidebar

---

## 📊 Domains Covered

The dataset covers papers from ArXiv in these areas:
- Machine Learning & Deep Learning
- Natural Language Processing
- Computer Vision
- Cyber Security
- Federated Learning
- Reinforcement Learning
- Transformers & Attention Mechanisms
- Artificial Intelligence
- Data Science

---

## 🔧 Configuration

You can customise the following in the sidebar:
- **Papers per query** (50–300): controls dataset size
- **Number of recommendations** (3–15): how many results to show

To add custom search domains, edit `DEFAULT_QUERIES` in `src/data_collector.py`.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.


