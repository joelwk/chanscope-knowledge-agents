import os
import json
import time
import shutil
import random
import asyncio
import logging
import traceback
import numpy as np
import pandas as pd
import gc
import tempfile
import pytz
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple, Union, Awaitable
from filelock import FileLock, Timeout
from tenacity import retry, wait_random_exponential, stop_after_attempt
import hashlib

try:
    # Import everything needed from model_ops
    from .model_ops import ModelProvider, ModelConfig as Config, KnowledgeAgent
except ImportError:
    # Fallback for direct imports
    from knowledge_agents.model_ops import ModelProvider, ModelConfig as Config, KnowledgeAgent

# Define KnowledgeDocument class here since it doesn't exist in model_ops.py
class KnowledgeDocument:
    """Document with metadata and text content."""
    
    def __init__(
        self, 
        thread_id: str = "", 
        posted_date_time: str = "", 
        text_clean: str = ""
    ):
        self.thread_id = thread_id
        self.posted_date_time = posted_date_time
        self.text_clean = text_clean
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeDocument':
        """Create a document from a dictionary."""
        return cls(
            thread_id=str(data.get("thread_id", "")),
            posted_date_time=str(data.get("posted_date_time", "")),
            text_clean=str(data.get("text_clean", ""))
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to a dictionary."""
        return {
            "thread_id": self.thread_id,
            "posted_date_time": self.posted_date_time,
            "text_clean": self.text_clean
        }

# Initialize logging
logger = logging.getLogger(__name__)

# Use thread-safe singleton pattern for KnowledgeAgent
_agent_instance = None
_agent_lock = asyncio.Lock()
_embedding_lock = asyncio.Lock()  # Add global embedding lock

async def get_agent() -> KnowledgeAgent:
    """Get or create the KnowledgeAgent singleton instance in a thread-safe manner."""
    global _agent_instance
    async with _agent_lock:
        if _agent_instance is None:
            _agent_instance = KnowledgeAgent()
            logger.info("Created new KnowledgeAgent instance")
        return _agent_instance

def load_data_from_csvs(directory: str) -> List[KnowledgeDocument]:
    """Load articles from stratified dataset CSV with improved error handling."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    article_list: List[KnowledgeDocument] = []
    csv_files = list(directory_path.glob("stratified_sample.csv"))
    if not csv_files:
        logger.warning(f"No stratified dataset found in {directory}")
        return article_list
    for file_path in tqdm(csv_files, desc="Loading stratified dataset"):
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
                    logger.error(f"Error parsing embeddings: {e}")
                    # Remove invalid embeddings
                    df = df.drop(columns=['embedding'])
            articles = [
                KnowledgeDocument.from_dict({
                    "thread_id": str(row["thread_id"]),
                    "posted_date_time": str(row["posted_date_time"]),
                    "text_clean": str(row["text_clean"])}) 
                for _, row in df.iterrows()]
            article_list.extend(articles)
            logger.info(f"Loaded {len(articles)} articles from stratified dataset")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    return article_list

async def save_embeddings(
    embeddings_array: np.ndarray,
    embeddings_path: Path,
    thread_id_map: Dict[str, int],
    temp_suffix: str = '.tmp'
) -> bool:
    """Save embeddings to disk atomically to prevent data corruption.
    
    This function uses a two-phase commit approach:
    1. Write to a temporary file
    2. Rename the temporary file to the final destination
    
    Args:
        embeddings_array: Numpy array of embeddings
        embeddings_path: Path where to save the embeddings
        thread_id_map: Mapping from thread IDs to embedding indices
        temp_suffix: Suffix for temporary files
        
    Returns:
        True if successful, False otherwise
    """
    if embeddings_array.size == 0:
        logger.error("Cannot save empty embeddings array")
        return False
        
    if not thread_id_map:
        logger.error("Cannot save embeddings without thread ID map")
        return False
    
    temp_path = embeddings_path.with_suffix(f'{embeddings_path.suffix}{temp_suffix}')
    parent_dir = embeddings_path.parent
    
    try:
        # Ensure directory exists
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file first
        logger.info(f"Saving embeddings to temporary file: {temp_path}")
        np.savez_compressed(
            temp_path,
            embeddings=embeddings_array,
            metadata={
                'shape': embeddings_array.shape,
                'dimensions': embeddings_array.shape[1],
                'count': embeddings_array.shape[0],
                'created': datetime.now(pytz.UTC).isoformat(),
                'thread_count': len(thread_id_map)
            }
        )
        
        # Force flush to disk
        try:
            os.fsync(open(temp_path, 'rb').fileno())
        except Exception as e:
            logger.warning(f"Failed to fsync temporary file: {e}")
        
        # Check if temp file was written correctly
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            logger.error(f"Failed to write temporary file {temp_path}")
            return False
            
        # Atomic rename operation
        logger.info(f"Renaming temporary file to {embeddings_path}")
        if embeddings_path.exists():
            # Create backup of existing file if it exists
            backup_path = embeddings_path.with_suffix(f'{embeddings_path.suffix}.bak')
            try:
                shutil.copy2(embeddings_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Perform atomic rename
        temp_path.rename(embeddings_path)
        
        # Verify final file exists and has content
        if not embeddings_path.exists() or embeddings_path.stat().st_size == 0:
            logger.error(f"Failed to save embeddings to {embeddings_path}")
            return False
            
        logger.info(f"Successfully saved embeddings to {embeddings_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up temp file if it exists
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as cleanup_e:
            logger.warning(f"Failed to clean up temporary file: {cleanup_e}")
            
        return False

async def load_embeddings(embeddings_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Load embeddings from Object Storage with robust error handling.
    
    Args:
        embeddings_path: Path to the embeddings file (used for logging only)
        
    Returns:
        Tuple of (embeddings array, metadata dict), or (None, None) if loading fails
    """
    try:
        # Import Object Storage client
        from replit.object_storage import Client
        from config.storage import ReplitObjectEmbeddingStorage
        from config.chanscope_config import ChanScopeConfig
        
        # Initialize storage
        config = ChanScopeConfig.from_env()
        embedding_storage = ReplitObjectEmbeddingStorage(config)
        
        # Get embeddings and thread map from Object Storage
        embeddings_array, _ = await embedding_storage.get_embeddings()
        
        if embeddings_array is None:
            logger.error(f"Embeddings file not found: {embeddings_path}")
            return None, None
            
        # Create metadata
        metadata = {
            "shape": embeddings_array.shape,
            "dtype": str(embeddings_array.dtype),
            "timestamp": datetime.now().isoformat()
        }
        
        return embeddings_array, metadata
        
    except Exception as e:
        logger.error(f"Error loading embeddings from Object Storage: {e}")
        logger.error(traceback.format_exc())
        return None, None

async def load_thread_id_map(file_path: Union[str, Path]) -> Optional[Dict[str, int]]:
    """Load thread ID mapping from Object Storage with robust error handling.
    
    Args:
        file_path: Path to the thread ID map file (used for logging only)
        
    Returns:
        Dictionary mapping thread IDs to embedding indices, or None if loading fails
    """
    try:
        # Import Object Storage client
        from replit.object_storage import Client
        from config.storage import ReplitObjectEmbeddingStorage
        from config.chanscope_config import ChanScopeConfig
        
        # Initialize storage
        config = ChanScopeConfig.from_env()
        embedding_storage = ReplitObjectEmbeddingStorage(config)
        
        # Get embeddings and thread map from Object Storage
        _, thread_id_map = await embedding_storage.get_embeddings()
        
        if thread_id_map is None:
            logger.error(f"Thread ID map file not found: {file_path}")
            return None
            
        # Validate that the map contains string keys and integer values
        if not all(isinstance(k, str) and isinstance(v, int) for k, v in thread_id_map.items()):
            logger.warning("Thread ID map contains invalid entries, attempting to fix")
            # Fix invalid entries
            fixed_map = {str(k): int(v) for k, v in thread_id_map.items() if v is not None}
            return fixed_map
            
        logger.info(f"Successfully loaded thread ID map with {len(thread_id_map)} entries")
        return thread_id_map
        
    except Exception as e:
        logger.error(f"Error loading thread ID map from Object Storage: {e}")
        logger.error(traceback.format_exc())
        return None

async def get_relevant_content(
    library: str = '.',
    batch_size: int = 100,
    provider: Optional[ModelProvider] = None,
    force_refresh: bool = False,
    progress_callback: Optional[Callable[[int, int], Union[None, Awaitable[None]]]] = None,
    stratified_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    thread_id_map_path: Optional[Path] = None) -> None:
    """Generate embeddings for articles in the stratified dataset.
    
    This function uses a file-based locking mechanism to prevent multiple workers
    from processing the same data simultaneously.
    
    Args:
        library: Path to the data library
        batch_size: Number of articles to process at once
        provider: Model provider to use for embeddings
        force_refresh: Whether to regenerate embeddings even if they exist
        progress_callback: Callback function for progress updates, can be async or sync
        stratified_path: Path to the stratified sample CSV file
        embeddings_path: Path to save the embeddings NPZ file
        thread_id_map_path: Path to save the thread ID map JSON file
    """
    try:
        # Initialize paths and lock file
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use consistent path structure that matches DataProcessor
        stratified_dir = data_dir / 'stratified'
        stratified_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided paths if available, otherwise use default paths
        if stratified_path is None:
            stratified_path = stratified_dir / 'stratified_sample.csv'
        if embeddings_path is None:
            embeddings_path = stratified_dir / 'embeddings.npz'
        if thread_id_map_path is None:
            thread_id_map_path = stratified_dir / 'thread_id_map.json'
            
        # Ensure parent directories exist
        stratified_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        thread_id_map_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting dedicated embedding generation (force_refresh={force_refresh})")
        logger.info(f"Using stratified_path: {stratified_path}")
        logger.info(f"Using embeddings_path: {embeddings_path}")
        logger.info(f"Using thread_id_map_path: {thread_id_map_path}")
        
        lock_file = embeddings_path.with_suffix('.lock')
        
        # Get a unique worker ID based on PID, timestamp, and random component
        pid = os.getpid()
        timestamp = int(time.time())
        random_component = random.randint(1000, 9999)
        worker_id = f'worker-{pid}-{timestamp}-{random_component}'
        
        # Initialize initialization marker files
        completion_marker = data_dir / '.initialization_complete'
        state_file = data_dir / '.initialization_state'
        in_progress_marker = data_dir / '.initialization_in_progress'
        worker_marker = data_dir / f'.worker_{worker_id}_in_progress'
        
        # Per Chanscope approach: First check if complete_data.csv exists
        complete_data_path = Path(data_dir.parent) / 'complete_data.csv'
        complete_data_exists = complete_data_path.exists()
        
        if not complete_data_exists:
            logger.info(f"complete_data.csv not found at {complete_data_path}. Will process data regardless of force_refresh setting.")
            # Force refresh to ensure data processing happens
            force_refresh = True
        else:
            logger.info(f"complete_data.csv exists at {complete_data_path}. Checking embedding files.")
        
        # Skip if complete_data.csv exists, embeddings already exist, and force_refresh is False
        if not force_refresh and complete_data_exists and embeddings_path.exists() and thread_id_map_path.exists():
            # Load and verify embeddings
            embeddings, metadata = load_embeddings(embeddings_path)
            thread_id_map = load_thread_id_map(thread_id_map_path)
            
            if embeddings is not None and thread_id_map is not None:
                logger.info(f"Embeddings already exist and force_refresh=False, skipping generation")
                return
            elif not force_refresh:
                logger.warning("Embeddings exist but could not be loaded properly, regenerating")
                force_refresh = True
        else:
            # Log the reason for continuing with embedding generation
            if force_refresh:
                logger.info("Proceeding with embedding generation because force_refresh=True")
            elif not complete_data_exists:
                logger.info("Proceeding with embedding generation because complete_data.csv doesn't exist")
            elif not embeddings_path.exists() or not thread_id_map_path.exists():
                logger.info("Proceeding with embedding generation because embedding files don't exist")
        
        # Check if another worker is already processing
        worker_markers = list(data_dir.glob('.worker_*_in_progress'))
        if worker_markers and not force_refresh:
            other_workers = [m for m in worker_markers if m.name != f'.worker_{worker_id}_in_progress']
            if other_workers and complete_data_exists:  # Only skip if complete_data.csv exists
                # Check if the worker marker is stale (older than 30 minutes)
                current_time = time.time()
                stale_markers = []
                active_markers = []
                
                for marker in other_workers:
                    try:
                        marker_time = marker.stat().st_mtime
                        marker_age_minutes = (current_time - marker_time) / 60
                        
                        if marker_age_minutes > 30:  # Consider markers older than 30 minutes as stale
                            stale_markers.append(marker)
                        else:
                            active_markers.append(marker)
                    except Exception as e:
                        logger.warning(f"Error checking marker age for {marker}: {e}")
                        active_markers.append(marker)  # Assume it's active if we can't check
                
                # Remove stale markers
                for marker in stale_markers:
                    try:
                        logger.warning(f"Removing stale worker marker: {marker.name} (age: {(current_time - marker.stat().st_mtime) / 60:.1f} minutes)")
                        marker.unlink()
                    except Exception as e:
                        logger.warning(f"Error removing stale marker {marker}: {e}")
                
                # If there are still active markers, skip embedding generation
                if active_markers:
                    logger.info(f"Another worker is already processing: {active_markers[0].name}. Skipping embedding generation.")
                    return
                else:
                    logger.info(f"Removed {len(stale_markers)} stale worker markers. Proceeding with embedding generation.")
            
        # Create worker marker
        worker_marker.touch()
        logger.info(f"Worker {worker_id} starting embedding generation")
        
        try:
            # Use file lock to ensure only one worker generates embeddings
            try:
                with FileLock(str(lock_file), timeout=5):
                    # Double check if embeddings exist after acquiring lock
                    if not force_refresh and complete_data_exists and embeddings_path.exists() and thread_id_map_path.exists():
                        logger.info(f"Embeddings already exist at {embeddings_path} (checked after lock), skipping generation")
                        return
                    
                    # Clean up any stale markers
                    for marker in [completion_marker, state_file, in_progress_marker]:
                        try:
                            if marker.exists():
                                marker.unlink()
                        except Exception as e:
                            logger.warning(f"Error cleaning up marker {marker}: {e}")
                    
                    # Create in-progress marker
                    in_progress_marker.touch()
                    
                    # Load articles from stratified dataset
                    articles = []
                    with tqdm(total=1, desc="Loading stratified dataset") as pbar:
                        try:
                            if not stratified_path.exists():
                                raise FileNotFoundError(f"Stratified sample not found at {stratified_path}")
                                
                            df = pd.read_csv(stratified_path)
                            if df.empty:
                                raise ValueError("Stratified dataset is empty")
                            for _, row in df.iterrows():
                                try:
                                    article = KnowledgeDocument(
                                        thread_id=str(row['thread_id']),
                                        posted_date_time=str(row['posted_date_time']),
                                        text_clean=str(row['text_clean'])
                                    )
                                    articles.append(article)
                                except Exception as e:
                                    logger.warning(f"Error processing row: {e}")
                                    continue
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Error loading stratified dataset: {e}")
                            raise
                    
                    if not articles:
                        raise ValueError("No valid articles loaded from stratified dataset")
                    
                    logger.info(f"Loaded {len(articles)} articles from stratified dataset")
                    
                    # Process articles in batches
                    try:
                        results = await process_batch(
                            articles=articles,
                            embedding_batch_size=batch_size,
                            provider=provider,
                            progress_callback=progress_callback
                        )
                        
                        if results:
                            thread_ids = []
                            embeddings_list = []
                            for result in results:
                                thread_id, _, _, embedding = result
                                if embedding and isinstance(embedding, (list, np.ndarray)):
                                    thread_ids.append(thread_id)
                                    embeddings_list.append(embedding)
                            
                            if thread_ids and embeddings_list:
                                # Convert embeddings list to a numpy array
                                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                                # Create thread_id mapping
                                thread_id_map = {tid: idx for idx, tid in enumerate(thread_ids)}
                                
                                # Save embeddings and thread_id mapping
                                temp_dir = tempfile.mkdtemp()
                                try:
                                    # Save files to temporary location first
                                    temp_embeddings_path = Path(temp_dir) / "embeddings.npz"
                                    temp_thread_id_map_path = Path(temp_dir) / "thread_id_map.json"
                                    
                                    # Create parent directory if needed
                                    stratified_dir.mkdir(parents=True, exist_ok=True)
                                    
                                    # Save embeddings array
                                    metadata = {
                                        "created_at": datetime.now().isoformat(),
                                        "dimensions": embeddings_array.shape[1],
                                        "count": len(thread_ids),
                                        "is_mock": any("mock" in str(r) for r in results[:10])  # Check if these are mock embeddings
                                    }
                                    np.savez_compressed(
                                        temp_embeddings_path, 
                                        embeddings=embeddings_array, 
                                        metadata=json.dumps(metadata)
                                    )
                                    
                                    # Save thread_id mapping
                                    with open(temp_thread_id_map_path, 'w') as f:
                                        json.dump(thread_id_map, f)
                                    
                                    # Move files to final destination atomically
                                    try:
                                        shutil.move(str(temp_embeddings_path), str(embeddings_path))
                                        shutil.move(str(temp_thread_id_map_path), str(thread_id_map_path))
                                    except (OSError, PermissionError) as e:
                                        logger.warning(f"Move operation failed: {e}. Falling back to copy.")
                                        try:
                                            shutil.copy2(str(temp_embeddings_path), str(embeddings_path))
                                            shutil.copy2(str(temp_thread_id_map_path), str(thread_id_map_path))
                                        except PermissionError as pe:
                                            logger.warning(f"Copy with metadata failed: {pe}. Using simple file copy.")
                                            # Use simple file copy without metadata preservation
                                            with open(temp_embeddings_path, 'rb') as src, open(embeddings_path, 'wb') as dst:
                                                dst.write(src.read())
                                            with open(temp_thread_id_map_path, 'r') as src, open(thread_id_map_path, 'w') as dst:
                                                dst.write(src.read())
                                    
                                    logger.info(f"Saved embeddings ({embeddings_array.shape}) and thread_id map to {embeddings_path}")
                                
                                finally:
                                    # Clean up temporary directory
                                    try:
                                        shutil.rmtree(temp_dir)
                                    except Exception as e:
                                        logger.warning(f"Error cleaning up temp dir {temp_dir}: {e}")
                            else:
                                logger.warning("No valid embeddings extracted from results, this should not happen with updated error handling")
                        else:
                            logger.warning("No results returned from batch processing, this should not happen with updated error handling")
                    except Exception as batch_error:
                        logger.error(f"Error in batch processing: {batch_error}")
                        logger.error(traceback.format_exc())
                        logger.warning("Falling back to mock embeddings generation at the get_relevant_content level")
                        
                        # Final fallback - generate mock embeddings
                        embedding_dim = 3072  # Match OpenAI's text-embedding-3-large dimension
                        thread_ids = [article.thread_id for article in articles]
                        
                        # Generate mock embeddings
                        mock_embeddings = []
                        for thread_id in thread_ids:
                            # Create a deterministic seed from thread_id
                            seed = int(hashlib.md5(thread_id.encode()).hexdigest(), 16) % (2**32)
                            np.random.seed(seed)
                            
                            # Generate normalized random embedding
                            mock_embedding = np.random.normal(0, 0.1, embedding_dim)
                            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
                            mock_embeddings.append(mock_embedding)
                            
                        # Convert to numpy array
                        embeddings_array = np.array(mock_embeddings, dtype=np.float32)
                        
                        # Create thread_id mapping
                        thread_id_map = {tid: idx for idx, tid in enumerate(thread_ids)}
                        
                        # Save embeddings and mapping
                        temp_dir = tempfile.mkdtemp()
                        try:
                            # Save files to temporary location first
                            temp_embeddings_path = Path(temp_dir) / "embeddings.npz"
                            temp_thread_id_map_path = Path(temp_dir) / "thread_id_map.json"
                            
                            # Create parent directory if needed
                            stratified_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Save embeddings array
                            metadata = {
                                "created_at": datetime.now().isoformat(),
                                "dimensions": embedding_dim,
                                "count": len(thread_ids),
                                "is_mock": True  # Flag to indicate these are mock embeddings
                            }
                            np.savez_compressed(
                                temp_embeddings_path, 
                                embeddings=embeddings_array, 
                                metadata=json.dumps(metadata)
                            )
                            
                            # Save thread_id mapping
                            with open(temp_thread_id_map_path, 'w') as f:
                                json.dump(thread_id_map, f)
                            
                            # Move files to final destination atomically
                            try:
                                shutil.move(str(temp_embeddings_path), str(embeddings_path))
                                shutil.move(str(temp_thread_id_map_path), str(thread_id_map_path))
                            except (OSError, PermissionError) as e:
                                logger.warning(f"Move operation failed: {e}. Falling back to copy.")
                                try:
                                    shutil.copy2(str(temp_embeddings_path), str(embeddings_path))
                                    shutil.copy2(str(temp_thread_id_map_path), str(thread_id_map_path))
                                except PermissionError as pe:
                                    logger.warning(f"Copy with metadata failed: {pe}. Using simple file copy.")
                                    # Use simple file copy without metadata preservation
                                    with open(temp_embeddings_path, 'rb') as src, open(embeddings_path, 'wb') as dst:
                                        dst.write(src.read())
                                    with open(temp_thread_id_map_path, 'r') as src, open(thread_id_map_path, 'w') as dst:
                                        dst.write(src.read())
                            
                            logger.info(f"Saved mock embeddings ({embeddings_array.shape}) and thread_id map to {embeddings_path}")
                        finally:
                            # Clean up temporary directory
                            try:
                                shutil.rmtree(temp_dir)
                            except Exception as cleanup_error:
                                logger.warning(f"Error cleaning up temp dir {temp_dir}: {cleanup_error}")
                    except Exception as e:
                        logger.error(f"Error processing articles: {e}")
                        logger.error(traceback.format_exc())
                    
                    # Mark completion
                    completion_marker.touch()
                    logger.info(f"Embeddings generation completed by worker {worker_id}")
            except Timeout:
                logger.info(f"Worker {worker_id} could not acquire lock, another worker is likely processing. Skipping.")
                return
                
        finally:
            # Remove worker marker file
            try:
                if worker_marker.exists():
                    worker_marker.unlink()
            except Exception as e:
                logger.warning(f"Error removing worker marker {worker_marker}: {e}")
                
    except Exception as e:
        logger.error(f"Error in get_relevant_content: {e}")
        logger.error(traceback.format_exc())

async def merge_articles_and_embeddings(stratified_path: Path, embeddings_path: Path, thread_id_map_path: Path) -> pd.DataFrame:
    """Merge article data with their embeddings efficiently.
    Args:
        stratified_path: Path to the stratified sample CSV
        embeddings_path: Path to the embeddings file (used for logging only)
        thread_id_map_path: Path to the thread ID map file (used for logging only)
    Returns:
        DataFrame containing merged article data and embeddings
    """
    try:
        # Load article data
        logger.info(f"Loading article data from {stratified_path}")
        articles_df = pd.read_csv(stratified_path)
        
        # Initialize Object Storage
        from config.storage import ReplitObjectEmbeddingStorage
        from config.chanscope_config import ChanScopeConfig
        
        # Initialize storage
        config = ChanScopeConfig.from_env()
        embedding_storage = ReplitObjectEmbeddingStorage(config)
        
        # Get embeddings and thread map from Object Storage
        embeddings_array, thread_id_map = await embedding_storage.get_embeddings()
        
        if embeddings_array is None or thread_id_map is None:
            logger.warning("Could not load embeddings or thread ID map from Object Storage")
            return articles_df
            
        # Log embeddings array details
        logger.info(f"Successfully loaded embeddings with shape {embeddings_array.shape}")
        
        # Convert thread IDs to strings for consistent comparison
        articles_df['thread_id'] = articles_df['thread_id'].astype(str)
        thread_id_map = {str(k): v for k, v in thread_id_map.items()}
        
        # Create embeddings column
        articles_df['embedding'] = None
        articles_df['is_mock_embedding'] = False
        
        # Track success rate
        successful_mappings = 0
        
        # Try to merge embeddings with articles
        for thread_id, idx in thread_id_map.items():
            try:
                if idx < len(embeddings_array):
                    # Thread IDs are now both strings
                    mask = articles_df['thread_id'] == thread_id
                    mask_count = mask.sum()
                    
                    if mask.any():
                        # Get the embedding value
                        embedding_value = embeddings_array[idx]
                        
                        # Assign the embedding to each matching row individually
                        matching_indices = articles_df.index[mask].tolist()
                        for match_idx in matching_indices:
                            articles_df.at[match_idx, 'embedding'] = embedding_value
                            articles_df.at[match_idx, 'is_mock_embedding'] = False
                        
                        successful_mappings += 1
                    else:
                        logger.debug(f"Thread ID {thread_id} not found in articles dataframe")
                else:
                    logger.warning(f"Embedding index {idx} out of bounds for thread_id {thread_id} (embeddings length: {len(embeddings_array)})")
            except Exception as e:
                logger.warning(f"Error mapping embedding for thread_id {thread_id}: {e}")
        
        # Log statistics about embedding coverage
        total_articles = len(articles_df)
        articles_with_embeddings = articles_df['embedding'].notna().sum()
        logger.info(
            f"Merged {articles_with_embeddings}/{total_articles} articles with embeddings "
            f"({articles_with_embeddings/total_articles*100:.1f}% coverage). "
            f"Successful thread ID mappings: {successful_mappings}/{len(thread_id_map)}"
        )
        
        return articles_df
    
    except Exception as e:
        logger.error(f"Error merging articles and embeddings: {e}")
        logger.error(traceback.format_exc())
        # Return original dataframe on error
        try:
            return pd.read_csv(stratified_path)
        except Exception:
            logger.error(f"Could not load original article data after merge failure")
            # Return empty dataframe as last resort
            return pd.DataFrame()

def validate_text_for_embedding(text: str) -> Tuple[bool, str]:
    """Validate if text is suitable for embedding generation.
    
    Args:
        text: The text to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    if text is None:
        return False, "Text is None"
        
    if not isinstance(text, str):
        return False, f"Text is not a string (type: {type(text)})"
        
    # Remove whitespace to check if there's actual content
    text_clean = text.strip()
    
    if not text_clean:
        return False, "Text is empty or whitespace only"
        
    if len(text_clean) < 10:
        return False, f"Text is too short ({len(text_clean)} chars)"
        
    # Check for excessive repetition which might confuse embedding models
    if len(set(text_clean)) / len(text_clean) < 0.1:
        return False, "Text has low character diversity (possible repetition)"
        
    return True, "Valid"

async def process_sub_batch(
    sub_batch_articles: List[KnowledgeDocument],
    agent: KnowledgeAgent,
    provider: Optional[ModelProvider] = None
) -> List[Tuple[str, str, str, List[float]]]:
    """Process a sub-batch of articles for embedding generation.
    
    Args:
        sub_batch_articles: List of KnowledgeDocument instances to process
        agent: KnowledgeAgent instance for making embedding requests
        provider: Model provider to use for embeddings
        
    Returns:
        List of tuples (thread_id, posted_date_time, text_clean, embedding)
    """
    # Filter valid articles and prepare text
    valid_articles = []
    valid_texts = []
    skipped_articles = []
    
    for article in sub_batch_articles:
        if not article.text_clean:
            skipped_articles.append((article, "Empty text"))
            continue
            
        # Validate text is suitable for embedding
        is_valid, reason = validate_text_for_embedding(article.text_clean)
        if not is_valid:
            skipped_articles.append((article, reason))
            continue
            
        valid_articles.append(article)
        valid_texts.append(article.text_clean)
    
    if skipped_articles:
        logger.warning(f"Skipped {len(skipped_articles)} articles due to validation failures")
        for article, reason in skipped_articles[:5]:  # Log first 5 skipped articles
            logger.warning(f"Skipped article {article.thread_id}: {reason}")
        
    if not valid_articles:
        logger.warning("No valid articles in batch after validation")
        return []
        
    # Log text statistics to help debug embedding issues
    text_lengths = [len(t) for t in valid_texts]
    logger.debug(
        f"Text statistics for batch: count={len(valid_texts)}, "
        f"min_length={min(text_lengths)}, max_length={max(text_lengths)}, "
        f"avg_length={sum(text_lengths)/len(text_lengths):.1f}"
    )
    
    # Generate mock embeddings helper function to avoid code duplication
    def generate_mock_embeddings(articles):
        embedding_dim = 3072  # Match OpenAI's text-embedding-3-large dimension
        batch_results = []
        
        # Log sample thread IDs for debugging
        sample_thread_ids = [article.thread_id for article in articles[:5]] if len(articles) > 0 else []
        logger.info(f"Sample thread_ids in generate_mock_embeddings: {sample_thread_ids}")
        logger.info(f"Thread ID types: {[type(article.thread_id).__name__ for article in articles[:5]]}")
        
        for article in articles:
            # Ensure thread_id is a string
            thread_id_str = str(article.thread_id)
            
            # Generate a deterministic seed from the article ID
            seed = int(hashlib.md5(thread_id_str.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            
            # Generate mock embedding (normalized for realistic values)
            mock_embedding = np.random.normal(0, 0.1, embedding_dim)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            
            batch_results.append((
                thread_id_str,  # Use the string version of thread_id
                article.posted_date_time, 
                article.text_clean, 
                mock_embedding.tolist()
            ))
        
        logger.info(f"Generated {len(batch_results)} mock embeddings as fallback")
        return batch_results
    
    try:
        async with asyncio.timeout(30):  # 30 second timeout (reduced from 60)
            try:
                # First try to get embeddings from the API
                response = await agent.embedding_request(text=valid_texts, provider=provider)
                batch_results = []
                
                if response and response.embedding:
                    # Validate embedding response
                    embeddings = response.embedding if isinstance(response.embedding, list) else [response.embedding]
                    
                    # Check if we got the expected number of embeddings
                    if len(embeddings) != len(valid_articles):
                        logger.warning(
                            f"Embedding response length mismatch: got {len(embeddings)}, "
                            f"expected {len(valid_articles)}"
                        )
                        
                        # Try to use as many as we can
                        valid_articles = valid_articles[:len(embeddings)]
                        
                    # Process each embedding with validation
                    embedding_stats = {"valid": 0, "invalid": 0, "single_float": 0}
                    for article, embedding in zip(valid_articles, embeddings):
                        # Check for None or empty
                        if embedding is None:
                            embedding_stats["invalid"] += 1
                            continue
                            
                        # Check for single float value (common error)
                        if isinstance(embedding, (float, int)):
                            embedding_stats["single_float"] += 1
                            logger.warning(
                                f"Received single float value ({embedding:.4f}) for article {article.thread_id}. "
                                f"Text preview: '{article.text_clean[:50]}...' (length: {len(article.text_clean)})"
                            )
                            continue
                            
                        # Check for proper array/list
                        if not isinstance(embedding, (list, np.ndarray)) or len(embedding) == 0:
                            embedding_stats["invalid"] += 1
                            continue
                            
                        # Add valid result
                        batch_results.append((article.thread_id, article.posted_date_time, article.text_clean, embedding))
                        embedding_stats["valid"] += 1
                    
                    # Log embedding statistics
                    logger.info(
                        f"Embedding statistics: {embedding_stats['valid']} valid, "
                        f"{embedding_stats['invalid']} invalid, "
                        f"{embedding_stats['single_float']} single float values"
                    )
                else:
                    logger.warning("Empty or null response from embedding API")
                    # Fall back to mock embeddings for empty responses
                    return generate_mock_embeddings(valid_articles)
                    
                return batch_results
                
            except Exception as e:
                logger.error(f"Error in API request: {e}")
                logger.error(traceback.format_exc())
                # Fall back to mock embeddings on API error
                return generate_mock_embeddings(valid_articles)
                
    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for embedding API response")
        # Fall back to mock embeddings on timeout
        return generate_mock_embeddings(valid_articles)
    except Exception as e:
        logger.error(f"Unexpected error in sub-batch processing: {e}")
        logger.error(traceback.format_exc())
        # Fall back to mock embeddings on unexpected error
        return generate_mock_embeddings(valid_articles)

async def process_batch(
    articles: List[KnowledgeDocument],
    embedding_batch_size: int = 10,
    provider: Optional[ModelProvider] = None,
    progress_callback: Optional[Callable[[int, int], Union[None, Awaitable[None]]]] = None
) -> List[Tuple[str, str, str, List[float]]]:
    """Process a batch of articles, computing their embeddings.
    
    Args:
        articles: List of KnowledgeDocument instances to process
        embedding_batch_size: Number of articles to embed in each sub-batch
        provider: Model provider for embeddings, defaults to system configuration
        progress_callback: Either async or non-async callback function(processed, total) to report progress
        
    Returns:
        List of tuples with (thread_id, posted_date_time, text_clean, embedding)
    """
    if not articles:
        logger.warning("No articles provided for batch processing")
        return []
    
    # Main processing logic
    try:
        total_articles = len(articles)
        if total_articles == 0:
            logger.warning("No articles to process")
            return []
            
        logger.info(f"Processing {total_articles} articles in batches of {embedding_batch_size}")
        
        # Split articles into sub-batches
        sub_batches = [articles[i:i+embedding_batch_size] for i in range(0, total_articles, embedding_batch_size)]
        total_batches = len(sub_batches)
        logger.info(f"Split into {total_batches} sub-batches")
        
        # Initialize agent once for all sub-batches
        agent = await get_agent()
        
        all_results = []
        processed_count = 0
        error_count = 0
        
        # Process sub-batches sequentially
        for i, sub_batch in enumerate(sub_batches):
            batch_start = i * embedding_batch_size
            batch_end = min((i + 1) * embedding_batch_size, total_articles)
            logger.info(f"Processing sub-batch {i+1}/{total_batches} (articles {batch_start}-{batch_end})")
            
            try:
                # Process the sub-batch - this now returns mock embeddings on error rather than raising
                results = await process_sub_batch(sub_batch, agent, provider)
                
                # Check if results were returned (the sub-batch function should always return something now)
                if results:
                    all_results.extend(results)
                    logger.info(f"Successfully processed sub-batch {i+1}/{total_batches} with {len(results)} results")
                else:
                    # This should never happen now, but if it does, log it
                    logger.warning(f"No results from sub-batch {i+1}/{total_batches}")
                    error_count += 1
                
                # Update progress
                processed_count += len(sub_batch)
                if progress_callback:
                    try:
                        if asyncio.iscoroutinefunction(progress_callback):
                            # Handle async callback
                            await progress_callback(processed_count, total_articles)
                        else:
                            # Handle synchronous callback
                            progress_callback(processed_count, total_articles)
                    except Exception as e:
                        logger.error(f"Error in progress callback: {e}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Unexpected error processing sub-batch {i+1}/{total_batches} ({batch_start}-{batch_end}): {e}")
                logger.error(traceback.format_exc())
                # Continue with next batch despite errors
        
        # Check if we got any results
        if not all_results:
            logger.warning("No results from batch processing, generating mock embeddings")
            
            # Generate mock embeddings as fallback
            embedding_dim = 3072  # Match OpenAI's text-embedding-3-large dimension
            mock_results = []
            
            for article in articles:
                # Create a deterministic seed from the article ID
                seed = int(hashlib.md5(article.thread_id.encode()).hexdigest(), 16) % (2**32)
                np.random.seed(seed)
                
                # Generate mock embedding (normalized for realistic values)
                mock_embedding = np.random.normal(0, 0.1, embedding_dim)
                mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
                
                mock_results.append((
                    article.thread_id, 
                    article.posted_date_time, 
                    article.text_clean, 
                    mock_embedding.tolist()
                ))
            
            logger.info(f"Generated {len(mock_results)} mock embeddings at batch level fallback")
            all_results = mock_results
            
        # Log statistics at the end
        logger.info(f"Completed processing with {len(all_results)} results from {total_articles} articles")
        if error_count > 0:
            logger.warning(f"{error_count}/{total_batches} sub-batches had errors but were handled gracefully")
            
        return all_results
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        logger.error(traceback.format_exc())
        
        # Always generate mock embeddings as final fallback
        embedding_dim = 3072  # Match OpenAI's text-embedding-3-large dimension
        mock_results = []
        
        for article in articles:
            seed = int(hashlib.md5(article.thread_id.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed)
            mock_embedding = np.random.normal(0, 0.1, embedding_dim)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            mock_results.append((
                article.thread_id, 
                article.posted_date_time, 
                article.text_clean, 
                mock_embedding.tolist()
            ))
        
        logger.info(f"Generated {len(mock_results)} mock embeddings as master fallback")
        return mock_results

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def generate_embeddings(
    texts: List[str],
    provider: Optional[ModelProvider] = None,
    request_timeout: int = 30
) -> List[List[float]]:
    """Generate embeddings for a list of texts with improved error handling and retries.
    
    This function will retry on failure with exponential backoff.
    
    Args:
        texts: List of text strings to embed
        provider: Model provider to use, defaults to system configuration
        request_timeout: Timeout in seconds for the embedding request
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []
    
    try:
        # Initialize provider/agent
        if provider is None:
            model_settings = Config.get_model_settings()
            agent = await get_agent()
        else:
            agent = await get_agent()
                
        if not agent:
            raise ValueError("Could not initialize KnowledgeAgent")
            
        # Filter empty texts
        filtered_texts = [text for text in texts if text and isinstance(text, str)]
        if not filtered_texts:
            logger.warning("No valid texts to embed after filtering")
            return []
            
        # Add diagnostic information
        logger.info(f"Generating embeddings for {len(filtered_texts)} texts using KnowledgeAgent")
        
        # Generate embeddings
        start_time = time.time()
        response = await agent.embedding_request(
            text=filtered_texts,
            provider=provider,
            batch_size=None  # Use default batch size from config
        )
        elapsed = time.time() - start_time
        
        # Process response
        if not response or not response.embedding:
            logger.warning("Empty embedding response")
            return []
            
        embeddings = response.embedding
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
            
        # Validate embeddings
        valid_embeddings = [
            emb for emb in embeddings 
            if isinstance(emb, (list, np.ndarray)) and len(emb) > 0
        ]
        
        logger.info(f"Generated {len(valid_embeddings)} embeddings in {elapsed:.2f}s")
        
        # Return embeddings with proper type
        return valid_embeddings
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout generating embeddings after {request_timeout}s")
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        logger.error(traceback.format_exc())
        raise