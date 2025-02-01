from .model_ops import KnowledgeAgent, load_config, ModelProvider
import logging
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

# Load configuration using centralized Config
model_config, app_config = load_config()

# Initialize agent with configuration
logger.info("Initializing KnowledgeAgent with configuration from settings...")
agent = KnowledgeAgent()

class Article:
    """Data class for article information."""
    def __init__(self, thread_id: str, posted_date_time: str, text_clean: str):
        self.thread_id = thread_id
        self.posted_date_time = posted_date_time
        self.text_clean = text_clean

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Article':
        return cls(
            thread_id=data["thread_id"],
            posted_date_time=data["posted_date_time"],
            text_clean=data["text_clean"])

def load_data_from_csvs(directory: str) -> List[Article]:
    """Load articles from all CSVs in a given directory with improved error handling."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    article_list: List[Article] = []
    csv_files = list(directory_path.glob("stratified_sample.csv"))

    if not csv_files:
        logger.warning(f"No stratified sample found in {directory}")
        return article_list

    for file_path in tqdm(csv_files, desc="Loading CSV files"):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            required_columns = {"thread_id", "posted_date_time", "text_clean"}

            if not required_columns.issubset(df.columns):
                logger.error(f"Missing required columns in {file_path}")
                continue

            # Convert embeddings from JSON if they exist
            if 'embedding' in df.columns:
                try:
                    df['embedding'] = df['embedding'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                except Exception as e:
                    logger.error(f"Error parsing embeddings: {str(e)}")
                    # Remove invalid embeddings
                    df = df.drop(columns=['embedding'])

            articles = [
                Article.from_dict({
                    "thread_id": str(row["thread_id"]),
                    "posted_date_time": str(row["posted_date_time"]),
                    "text_clean": str(row["text_clean"])
                }) 
                for _, row in df.iterrows()
            ]
            article_list.extend(articles)
            logger.info(f"Loaded {len(articles)} articles from {file_path}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    return article_list

async def get_relevant_content(
    library: str = '.',
    knowledge_base: str = '.',
    batch_size: int = 100,
    provider: Optional[ModelProvider] = None,
    force_refresh: bool = False
) -> None:
    """Create knowledge base with embeddings from articles in library."""
    logger.info("Creating knowledge base with embeddings...")

    # Check if knowledge base already exists and is valid
    kb_path = Path(knowledge_base)
    if not force_refresh and kb_path.exists():
        try:
            # Verify the knowledge base is valid
            df = pd.read_csv(kb_path)
            if 'embedding' in df.columns and len(df) > 0:
                logger.info("Using existing knowledge base with embeddings")
                return
        except Exception as e:
            logger.warning(f"Existing knowledge base invalid, will recreate: {e}")

    # Load articles from CSV files
    articles = load_data_from_csvs(library)
    if not articles:
        logger.warning("No articles found in the library.")
        return

    # Calculate optimal batch size based on total articles
    def get_optimal_batch_size(count: int) -> int:
        if count < 1000:
            return 100
        elif count < 5000:
            return 75
        elif count < 10000:
            return 50
        else:
            return 25
    
    # Determine optimal batch size based on number of articles
    article_count = len(articles)
    optimal_batch_size = get_optimal_batch_size(article_count)
    
    if batch_size != optimal_batch_size:
        logger.info(f"Adjusting batch size from {batch_size} to {optimal_batch_size} based on article count")
        batch_size = optimal_batch_size

    logger.info(f"Starting embedding generation for {len(articles)} articles with batch size {batch_size}")
    total_batches = len(articles) // batch_size + (1 if len(articles) % batch_size else 0)
    logger.info(f"Will process {total_batches} batches")

    try:
        # Process articles in batches
        results = await process_article_batch(
            articles=articles,
            embedding_batch_size=batch_size,
            provider=provider
        )

        # Save results to knowledge base
        if results:
            logger.info(f"Creating DataFrame with {len(results)} articles and embeddings")
            df = pd.DataFrame(
                results,
                columns=['thread_id', 'posted_date_time', 'text_clean', 'embedding']
            )
            # Ensure embeddings are valid before converting to JSON
            df['embedding'] = df['embedding'].apply(lambda x: json.dumps(x) if isinstance(x, (list, np.ndarray)) else None)
            # Remove rows with invalid embeddings
            df = df.dropna(subset=['embedding'])
            df.to_csv(knowledge_base, index=False)
            logger.info(f"Saved {len(df)} articles with embeddings to knowledge base")
        else:
            logger.warning("No results to save to knowledge base")

    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def process_article_batch(
    articles: List[Article],
    embedding_batch_size: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[List]:
    """Process a batch of articles to get embeddings using the specified provider."""
    results = []
    total_batches = len(articles) // embedding_batch_size + (1 if len(articles) % embedding_batch_size else 0)
    logger.info(f"Processing {len(articles)} articles in {total_batches} batches")

    # Use tqdm for progress tracking
    progress_bar = tqdm(range(0, len(articles), embedding_batch_size), 
                       desc="Processing batches",
                       total=total_batches)

    # Process in batches according to the specified embedding_batch_size
    for i in progress_bar:
        batch = articles[i:i + embedding_batch_size]
        current_batch = i // embedding_batch_size + 1
        
        try:
            # Get embeddings for all texts in batch
            texts = [str(article.text_clean).strip() for article in batch]
            # Filter out empty or invalid texts
            valid_texts = []
            valid_articles = []
            for text, article in zip(texts, batch):
                if text and len(text.strip()) > 0:
                    valid_texts.append(text)
                    valid_articles.append(article)

            if not valid_texts:
                logger.warning(f"No valid texts in batch {current_batch}")
                continue

            logger.info(f"Batch {current_batch}: Processing {len(valid_texts)} valid texts")
            
            try:
                # Make a single embedding request for the batch
                response = await agent.embedding_request(
                    text=valid_texts,
                    provider=provider,
                    batch_size=embedding_batch_size
                )

                if not hasattr(response, 'embedding'):
                    error_msg = f"Invalid response format from provider {provider}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Create results only for valid texts
                embeddings = response.embedding if isinstance(response.embedding, list) else [response.embedding]
                for article, embedding in zip(valid_articles, embeddings):
                    results.append([
                        article.thread_id,
                        article.posted_date_time,
                        article.text_clean,
                        embedding,
                    ])
                
                # Update progress bar description
                progress_bar.set_description(f"Processed {len(results)}/{len(articles)} articles")
                
            except Exception as embed_err:
                logger.error(f"Embedding request failed for batch {current_batch}: {str(embed_err)}")
                raise RuntimeError(f"Embedding request failed: {str(embed_err)}") from embed_err
        except Exception as e:
            logger.error(f"Error processing batch {current_batch}: {type(e).__name__}: {str(e)}")
            raise

    logger.info(f"Completed processing all batches. Generated {len(results)} embeddings")
    return results