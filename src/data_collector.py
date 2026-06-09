"""
Data Collector Module
Fetches research papers from the ArXiv API and stores them in CSV format.
"""

import arxiv
import pandas as pd
import time
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Default search queries covering the target domains
DEFAULT_QUERIES = [
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "cyber security",
    "data science",
    "artificial intelligence",
    "transformer neural network",
    "reinforcement learning",
    "federated learning",
]

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.csv")


def fetch_papers(
    queries: list[str] = None,
    max_per_query: int = 150,
    output_path: str = DATA_PATH,
) -> pd.DataFrame:
    """
    Fetch research papers from ArXiv for the given queries.

    Args:
        queries: List of search query strings.
        max_per_query: Maximum papers to fetch per query.
        output_path: Path where the CSV will be saved.

    Returns:
        DataFrame containing all collected papers (deduplicated).
    """
    if queries is None:
        queries = DEFAULT_QUERIES

    all_papers = []
    seen_ids = set()

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)

    for query in queries:
        logger.info(f"Fetching papers for query: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        try:
            results = list(client.results(search))
            count = 0
            for paper in tqdm(results, desc=f"  {query}", leave=False):
                paper_id = paper.entry_id
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                all_papers.append(
                    {
                        "id": paper_id,
                        "title": paper.title.strip(),
                        "abstract": paper.summary.strip(),
                        "authors": ", ".join(str(a) for a in paper.authors),
                        "categories": ", ".join(paper.categories),
                        "published": str(paper.published.date()) if paper.published else "",
                        "pdf_url": paper.pdf_url or "",
                        "query": query,
                    }
                )
                count += 1

            logger.info(f"  Collected {count} unique papers for '{query}'")
            time.sleep(1)

        except Exception as e:
            logger.error(f"  Error fetching '{query}': {e}")
            continue

    if not all_papers:
        logger.warning("No papers collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_papers)
    df.drop_duplicates(subset=["id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} papers to '{output_path}'")

    return df


def load_papers(path: str = DATA_PATH) -> pd.DataFrame:
    """Load papers CSV. Returns empty DataFrame if file not found."""
    if not os.path.exists(path):
        logger.warning(f"Dataset not found at '{path}'. Run fetch_papers() first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df


if __name__ == "__main__":
    df = fetch_papers(max_per_query=200)
    print(f"\nTotal papers collected: {len(df)}")
    print(df.head())
