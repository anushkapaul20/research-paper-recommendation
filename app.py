"""
Research Paper Recommendation System
Streamlit UI — upload a PDF or enter a query to find similar papers.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
import sys
import logging
from collections import Counter

# Add project root to path so `src` imports work
sys.path.insert(0, os.path.dirname(__file__))

from src.recommender import recommend_from_bytes, recommend_from_text, load_or_build_index
from src.data_collector import fetch_papers, load_papers
from src.vector_store import get_store

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .paper-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .paper-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #cdd6f4;
    }
    .score-badge {
        background: #a6e3a1;
        color: #1e1e2e;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .tag {
        background: #313244;
        color: #cba6f7;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #89b4fa;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — dataset management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("📦 Dataset")

    data_path = os.path.join(os.path.dirname(__file__), "data", "papers.csv")
    index_ready = get_store().is_ready()

    # Check if CSV exists
    csv_exists = os.path.exists(data_path)
    if csv_exists:
        df_check = pd.read_csv(data_path)
        st.success(f"✅ Dataset loaded: **{len(df_check):,}** papers")
    else:
        st.warning("⚠️ No dataset found.")

    # Check index
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    index_file = os.path.join(models_dir, "faiss.index")
    if os.path.exists(index_file):
        st.success("✅ FAISS index ready")
    else:
        st.warning("⚠️ FAISS index not built yet")

    st.divider()
    st.subheader("🔄 Build / Refresh")

    max_per_query = st.slider("Papers per query", 50, 300, 150, 50)

    if st.button("🚀 Fetch Papers & Build Index", use_container_width=True):
        with st.spinner("Fetching papers from ArXiv… this may take a few minutes."):
            try:
                df = fetch_papers(max_per_query=max_per_query)
                st.success(f"Fetched {len(df):,} papers.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")
                st.stop()

        with st.spinner("Building FAISS index…"):
            try:
                from src.recommender import build_index_from_csv
                build_index_from_csv(data_path)
                st.success("Index built successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Index build failed: {e}")

    if csv_exists and not os.path.exists(index_file):
        if st.button("🔨 Build Index from Existing CSV", use_container_width=True):
            with st.spinner("Building FAISS index…"):
                try:
                    from src.recommender import build_index_from_csv
                    build_index_from_csv(data_path)
                    st.success("Index built!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Index build failed: {e}")

    st.divider()
    top_k = st.slider("Number of recommendations", 3, 15, 5)
    st.divider()
    st.caption("Model: all-MiniLM-L6-v2 · Index: FAISS (cosine similarity)")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("📄 Research Paper Recommendation System")
st.markdown("Upload your paper as a **PDF** or enter a **text query** to discover related research.")

# Load index on startup (non-blocking)
if not get_store().is_ready():
    with st.spinner("Loading index…"):
        load_or_build_index()

# ---------------------------------------------------------------------------
# Input tabs
# ---------------------------------------------------------------------------
tab_pdf, tab_text, tab_analytics = st.tabs(["📎 Upload PDF", "🔍 Text Query", "📊 Dataset Analytics"])

results = None
mode = None

with tab_pdf:
    uploaded_file = st.file_uploader(
        "Upload a research paper PDF",
        type=["pdf"],
        help="The system extracts the abstract and finds the most similar papers.",
    )
    if uploaded_file:
        if st.button("🔎 Find Similar Papers", key="pdf_btn", use_container_width=True):
            with st.spinner("Analysing PDF…"):
                pdf_bytes = uploaded_file.read()
                results = recommend_from_bytes(pdf_bytes, top_k=top_k)
                mode = "pdf"

with tab_text:
    query_text = st.text_area(
        "Enter a title, abstract, or research description",
        height=150,
        placeholder="e.g. Federated learning for anomaly detection in IoT networks using differential privacy…",
    )
    if st.button("🔎 Find Similar Papers", key="text_btn", use_container_width=True):
        if not query_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Searching…"):
                results = recommend_from_text(query_text, top_k=top_k)
                mode = "text"

# ---------------------------------------------------------------------------
# Analytics tab
# ---------------------------------------------------------------------------
with tab_analytics:
    data_path = os.path.join(os.path.dirname(__file__), "data", "papers.csv")
    if not os.path.exists(data_path):
        st.info("👈 No dataset found. Use the sidebar to **Fetch Papers & Build Index** first.")
    else:
        df_analytics = pd.read_csv(data_path)
        df_analytics.fillna("", inplace=True)

        st.markdown("### 📊 Dataset Overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Papers", f"{len(df_analytics):,}")
        m2.metric("Unique Authors", f"{len(set(a.strip() for authors in df_analytics['authors'] for a in authors.split(',') if a.strip())):,}")
        year_counts = pd.to_datetime(df_analytics["published"], errors="coerce").dt.year
        m3.metric("Year Range", f"{int(year_counts.min())} – {int(year_counts.max())}" if year_counts.notna().any() else "N/A")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Top categories bar chart ──────────────────────────────────────
        with col1:
            st.markdown("**📂 Top ArXiv Categories**")
            all_cats = []
            for cats in df_analytics["categories"]:
                for c in cats.split(","):
                    c = c.strip()
                    if c:
                        all_cats.append(c)
            cat_counts = Counter(all_cats).most_common(12)
            cat_df = pd.DataFrame(cat_counts, columns=["Category", "Count"])

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            fig1.patch.set_facecolor("#1e1e2e")
            ax1.set_facecolor("#1e1e2e")
            bars = ax1.barh(cat_df["Category"][::-1], cat_df["Count"][::-1], color="#89b4fa")
            ax1.set_xlabel("Number of Papers", color="#cdd6f4")
            ax1.tick_params(colors="#cdd6f4", labelsize=8)
            ax1.spines[:].set_color("#313244")
            for bar in bars:
                ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                         str(int(bar.get_width())), va="center", color="#cdd6f4", fontsize=7)
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

        # ── Papers per year line chart ────────────────────────────────────
        with col2:
            st.markdown("**📅 Papers Published per Year**")
            df_analytics["year"] = pd.to_datetime(df_analytics["published"], errors="coerce").dt.year
            year_series = df_analytics["year"].dropna().astype(int).value_counts().sort_index()

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor("#1e1e2e")
            ax2.set_facecolor("#1e1e2e")
            ax2.plot(year_series.index, year_series.values, color="#a6e3a1", linewidth=2, marker="o", markersize=5)
            ax2.fill_between(year_series.index, year_series.values, alpha=0.2, color="#a6e3a1")
            ax2.set_xlabel("Year", color="#cdd6f4")
            ax2.set_ylabel("Papers", color="#cdd6f4")
            ax2.tick_params(colors="#cdd6f4", labelsize=8)
            ax2.spines[:].set_color("#313244")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.divider()
        col3, col4 = st.columns(2)

        # ── Top search queries ────────────────────────────────────────────
        with col3:
            st.markdown("**🔎 Papers per Search Query**")
            if "query" in df_analytics.columns:
                query_counts = df_analytics["query"].value_counts().head(10)
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                fig3.patch.set_facecolor("#1e1e2e")
                ax3.set_facecolor("#1e1e2e")
                ax3.barh(query_counts.index[::-1], query_counts.values[::-1], color="#cba6f7")
                ax3.set_xlabel("Count", color="#cdd6f4")
                ax3.tick_params(colors="#cdd6f4", labelsize=8)
                ax3.spines[:].set_color("#313244")
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("Query column not available.")

        # ── Abstract length distribution ──────────────────────────────────
        with col4:
            st.markdown("**📝 Abstract Length Distribution**")
            df_analytics["abstract_len"] = df_analytics["abstract"].str.split().str.len()
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            fig4.patch.set_facecolor("#1e1e2e")
            ax4.set_facecolor("#1e1e2e")
            ax4.hist(df_analytics["abstract_len"].dropna(), bins=30, color="#f38ba8", edgecolor="#1e1e2e")
            ax4.set_xlabel("Word Count", color="#cdd6f4")
            ax4.set_ylabel("Papers", color="#cdd6f4")
            ax4.tick_params(colors="#cdd6f4", labelsize=8)
            ax4.spines[:].set_color("#313244")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close(fig4)

        # ── Top prolific authors ──────────────────────────────────────────
        st.divider()
        st.markdown("**👩‍🔬 Top 10 Most Prolific Authors in Dataset**")
        author_counter: Counter = Counter()
        for authors_str in df_analytics["authors"]:
            for a in authors_str.split(","):
                a = a.strip()
                if a:
                    author_counter[a] += 1
        top_authors = pd.DataFrame(author_counter.most_common(10), columns=["Author", "Papers"])
        st.dataframe(top_authors, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if results:
    if "error" in results:
        st.error(results["error"])
        st.info("👈 Use the sidebar to fetch papers and build the index first.")
        st.stop()

    st.divider()

    # ── Keywords ──────────────────────────────────────────────────────────
    if results.get("keywords"):
        st.markdown('<div class="section-header">🏷️ Extracted Keywords</div>', unsafe_allow_html=True)
        kw_html = " ".join(f'<span class="tag">{kw}</span>' for kw in results["keywords"])
        st.markdown(kw_html, unsafe_allow_html=True)

    # ── Abstract snippet (PDF mode) ────────────────────────────────────────
    if mode == "pdf" and results.get("abstract_snippet"):
        with st.expander("📝 Extracted Abstract Snippet"):
            st.write(results["abstract_snippet"])

    # ── Similar papers ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📚 Similar Papers</div>', unsafe_allow_html=True)

    papers = results.get("similar_papers", [])
    if not papers:
        st.info("No similar papers found.")
    else:
        for i, paper in enumerate(papers, 1):
            title   = paper.get("title", "Untitled")
            authors = paper.get("authors", "Unknown")[:120]
            abstract = paper.get("abstract", "")[:300]
            score   = paper.get("similarity_pct", "—")
            cats    = paper.get("categories", "")
            pub     = paper.get("published", "")
            pdf_url = paper.get("pdf_url", "")

            cat_tags = " ".join(
                f'<span class="tag">{c.strip()}</span>'
                for c in cats.split(",")[:3] if c.strip()
            )

            pdf_link = (
                f'<a href="{pdf_url}" target="_blank" style="color:#89b4fa;font-size:0.85rem;">📥 PDF</a>'
                if pdf_url else ""
            )

            st.markdown(
                f"""
                <div class="paper-card">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <span class="paper-title">{i}. {title}</span>
                    <span class="score-badge">{score}</span>
                  </div>
                  <div style="color:#a6adc8;font-size:0.85rem;margin:6px 0;">👥 {authors}</div>
                  <div style="color:#9399b2;font-size:0.83rem;margin-bottom:8px;">{abstract}…</div>
                  <div>{cat_tags} &nbsp; {pdf_link} &nbsp;
                    <span style="color:#585b70;font-size:0.78rem;">📅 {pub}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Two-column: Authors + Directions ───────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">👩‍🔬 Recommended Authors</div>', unsafe_allow_html=True)
        authors_list = results.get("authors", [])
        if authors_list:
            for author in authors_list:
                st.markdown(f"- {author}")
        else:
            st.info("No author data available.")

    with col_b:
        st.markdown('<div class="section-header">🧭 Suggested Research Directions</div>', unsafe_allow_html=True)
        directions = results.get("directions", [])
        if directions:
            for d in directions:
                st.markdown(f"- {d}")
        else:
            st.info("No directions inferred.")

    # ── Export results ──────────────────────────────────────────────────────
    st.divider()
    if papers:
        export_df = pd.DataFrame(papers)[
            [c for c in ["title", "authors", "published", "categories", "similarity_pct", "pdf_url"]
             if c in pd.DataFrame(papers).columns]
        ]
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Results as CSV",
            data=csv_data,
            file_name="recommendations.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
elif not get_store().is_ready():
    st.info("👈 No index loaded yet. Use the sidebar to **Fetch Papers & Build Index** first.")
