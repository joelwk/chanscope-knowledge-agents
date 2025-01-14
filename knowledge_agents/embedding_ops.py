from csv import writer
from .model_ops import KnowledgeAgent, load_config, ModelProvider
import logging
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            text_clean=data["text_clean"]
        )

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
    provider: Optional[ModelProvider] = None
) -> None:
    """Create knowledge base with embeddings from articles in library."""
    logger.info("Creating knowledge base with embeddings...")
    
    # Load articles from CSV files
    articles = load_data_from_csvs(library)
    if not articles:
        logger.warning("No articles found in the library.")
        return
    
    try:
        # Process articles in batches
        results = await process_article_batch(
            articles=articles,
            batch_size=batch_size,
            provider=provider
        )
        
        # Save results to knowledge base
        if results:
            logger.info(f"Creating DataFrame with {len(results)} articles and embeddings")
            df = pd.DataFrame(
                results,
                columns=['thread_id', 'posted_date_time', 'text_clean', 'embedding']
            )
            # Convert embeddings to strings for CSV storage
            df['embedding'] = df['embedding'].apply(json.dumps)
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
    batch_size: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[List]:
    """Process a batch of articles to get embeddings using the specified provider.
    
    This function should only be used with embedding-specific models.
    """
    results = []
    
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        try:
            # Get embeddings for all texts in batch at once
            texts = [article.text_clean for article in batch]
            response = await agent.embedding_request(texts, provider)
            
            # Create results
            for article, embedding in zip(batch, response.embedding):
                results.append([
                    article.thread_id,
                    article.posted_date_time,
                    article.text_clean,
                    embedding,
                ])
                
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
            raise
            
    return results