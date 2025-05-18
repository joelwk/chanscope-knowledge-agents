import ast
import asyncio
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tiktoken
from scipy import spatial

from .model_ops import KnowledgeAgent, ModelOperation, ModelProvider

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def prepare_embeddings(
    df: pd.DataFrame,
    query_embedding: Optional[np.ndarray] = None,
    adapt: bool = False,
) -> Tuple[np.ndarray, List[int], Optional[np.ndarray]]:
    """Validate and normalize embeddings from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an ``embedding`` column.
    query_embedding : Optional[np.ndarray]
        Embedding for the query used to check dimensions. If ``None`` only
        validation is performed.
    adapt : bool
        When ``True`` and no embeddings match ``query_embedding``'s dimension,
        attempt dimension adaptation using padding/truncation.

    Returns
    -------
    Tuple[np.ndarray, List[int], Optional[np.ndarray]]
        Normalized embedding array, indices of valid rows, and possibly
        modified query embedding if adaptation was required.
    """

    if "embedding" not in df.columns:
        logger.error("No embeddings found in DataFrame")
        raise ValueError("DataFrame must contain 'embedding' column")

    embeddings: List[Union[np.ndarray, List[float]]] = []
    valid_indices: List[int] = []

    for i, emb in enumerate(df["embedding"].values):
        if isinstance(emb, str):
            try:
                parsed_emb = json.loads(emb)
                if parsed_emb:
                    embeddings.append(parsed_emb)
                    valid_indices.append(i)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse embedding JSON: {emb[:100]}...")
                continue
        elif isinstance(emb, list):
            if not emb:
                continue
            embeddings.append(emb)
            valid_indices.append(i)
        elif isinstance(emb, np.ndarray):
            if emb.size == 0:
                continue
            if emb.size == 1:
                logger.warning(f"Converting single-item array at index {i} to vector")
                embeddings.append([float(emb.item()), 0.0, 0.0])
            else:
                embeddings.append(emb)
            valid_indices.append(i)
        elif isinstance(emb, (float, int)):
            logger.warning(f"Converting single float value at index {i} to array")
            embeddings.append([float(emb), 0.0, 0.0])
            valid_indices.append(i)
        else:
            logger.warning(f"Unexpected embedding format at index {i}: {type(emb)}")

    if not embeddings:
        raise ValueError("No valid embeddings found in DataFrame")

    prepared_query = query_embedding

    if query_embedding is not None:
        embedding_dimensions = [len(e) if hasattr(e, "__len__") else 1 for e in embeddings]
        if len(set(embedding_dimensions)) > 1:
            logger.warning(f"Mixed embedding dimensions found: {set(embedding_dimensions)}")
            query_dim = len(query_embedding) if hasattr(query_embedding, "__len__") else 1
            filtered_embeddings: List[Union[np.ndarray, List[float]]] = []
            filtered_indices: List[int] = []
            for emb, orig_idx in zip(embeddings, valid_indices):
                emb_dim = len(emb) if hasattr(emb, "__len__") else 1
                if emb_dim == query_dim:
                    filtered_embeddings.append(emb)
                    filtered_indices.append(orig_idx)

            if not filtered_embeddings:
                if not adapt:
                    raise ValueError(f"No embeddings with matching dimension {query_dim} found")

                logger.warning(
                    f"No embeddings with matching dimension {query_dim} found, using dimension adaptation"
                )

                substantial_embeddings: List[np.ndarray] = []
                substantial_indices: List[int] = []
                for emb, orig_idx in zip(embeddings, valid_indices):
                    if hasattr(emb, "__len__") and len(emb) > 3:
                        substantial_embeddings.append(np.array(emb, dtype=np.float32))
                        substantial_indices.append(orig_idx)

                if substantial_embeddings:
                    embeddings_arr = np.vstack(substantial_embeddings)
                    min_dim = min(embeddings_arr.shape[1], len(query_embedding))
                    prepared_query = query_embedding[:min_dim]
                    embeddings = embeddings_arr[:, :min_dim]
                    valid_indices = substantial_indices
                else:
                    processed_embeddings = []
                    for emb in embeddings:
                        if hasattr(emb, "__len__"):
                            curr_len = len(emb)
                            if curr_len < query_dim:
                                padded = np.zeros(query_dim, dtype=np.float32)
                                padded[:curr_len] = emb
                                processed_embeddings.append(padded)
                            else:
                                processed_embeddings.append(
                                    np.array(emb[:query_dim], dtype=np.float32)
                                )
                        else:
                            padded = np.zeros(query_dim, dtype=np.float32)
                            padded[0] = float(emb)
                            processed_embeddings.append(padded)
                    embeddings = np.vstack(processed_embeddings)
            else:
                embeddings = filtered_embeddings
                valid_indices = filtered_indices

    if not isinstance(embeddings, np.ndarray):
        try:
            embeddings = np.vstack(embeddings)
        except ValueError as e:
            logger.error(f"Failed to stack embeddings: {e}")
            embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
            embeddings = np.vstack(embeddings)

    return embeddings, valid_indices, prepared_query


async def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    agent: KnowledgeAgent,
    top_n: int = 50,
    provider: Optional[ModelProvider] = None,
    use_recursive: bool = False,
    final_top_n: int = 10,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Find strings most related to a query using embeddings.

    Args:
        query: The query string to find related content for
        df: DataFrame containing text and embeddings
        agent: KnowledgeAgent instance for embedding generation
        top_n: Number of top results to return
        provider: Optional model provider to use for embeddings
        use_recursive: Whether to use recursive refinement (default: False)
        final_top_n: Final number of results if using recursive approach

    Returns:
        List of tuples containing (text, similarity_score, metadata)
    """
    # If recursive approach is requested, delegate to that function
    if use_recursive:
        return await recursive_strings_ranked_by_relatedness(
            query=query,
            df=df,
            agent=agent,
            final_top_n=final_top_n,
            initial_top_n=top_n,
            provider=provider,
        )

    # Original implementation for non-recursive approach
    try:
        query_embedding_response = await agent.embedding_request(text=query, provider=provider)
        query_embedding = query_embedding_response.embedding

        embeddings, valid_indices, query_embedding = prepare_embeddings(
            df=df,
            query_embedding=query_embedding,
            adapt=True,
        )

        similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric="cosine")[0]
        
        # Get indices of top N similar items
        # Ensure we don't take more than available
        actual_top_n = min(top_n, len(similarities))
        top_similarity_indices = np.argsort(similarities)[::-1][:actual_top_n]

        # Map back to original dataframe indices
        top_df_indices = [valid_indices[i] for i in top_similarity_indices]

        # Return relevant records with similarity scores
        results = []
        for i, idx in enumerate(top_df_indices):
            sim_score = similarities[top_similarity_indices[i]]

            # Get the text content - prioritize text_clean but fall back to content column
            # This ensures compatibility with both S3 data (text_clean) and PostgreSQL data (content)
            if "text_clean" in df.columns and pd.notna(df.iloc[idx].get("text_clean", "")):
                text_content = df.iloc[idx]["text_clean"]
            elif "content" in df.columns and pd.notna(df.iloc[idx].get("content", "")):
                text_content = df.iloc[idx]["content"]
            else:
                # If neither column exists or both are null, use an empty string
                text_content = ""
                logger.warning(
                    f"Row {idx} has no valid text content in either text_clean or content columns"
                )

            # Create result tuple with text content and metadata
            results.append(
                (
                    text_content,
                    float(sim_score),
                    {
                        "thread_id": str(df.iloc[idx]["thread_id"]),
                        "posted_date_time": str(df.iloc[idx]["posted_date_time"]),
                        "similarity_score": float(sim_score),
                    },
                )
            )

        return results[:final_top_n]
        
    except Exception as e:
        logger.error(f"Error ranking strings by relatedness: {e}")
        traceback.print_exc()
        raise

def is_valid_chunk(text: str) -> bool:
    """Check if a chunk has enough meaningful content."""
    # Remove XML tags and whitespace
    content = text.split("<content>")[-1].split("</content>")[0].strip()
    # Minimum requirements
    min_words = 5
    min_chars = 20
    words = [w for w in content.split() if len(w) > 2]  # Filter out very short words
    return len(words) >= min_words and len(content) >= min_chars


def clean_chunk_text(text: str) -> str:
    """Clean and format chunk text to remove excessive whitespace."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and not line.isspace():
            if any(
                tag in line
                for tag in ["<temporal_context>", "</temporal_context>", "<content>", "</content>"]
            ):
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.strip())

    return "\n".join(cleaned_lines)

def create_chunks(text: str, n: int, tokenizer=None) -> List[Dict[str, Any]]:
    """Creates chunks of text, preserving sentence boundaries where possible."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # First, split text into sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Add period back to sentence
        sentence = sentence + "."
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)

        # If single sentence is too long, split it
        if sentence_length > n:
            # If we have a current chunk, add it first
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({"text": chunk_text, "token_count": current_length})
                current_chunk = []
                current_length = 0

            # Split long sentence into chunks
            words = sentence.split()
            temp_chunk = []
            temp_length = 0

            for word in words:
                word_tokens = tokenizer.encode(word + " ")
                if temp_length + len(word_tokens) > n:
                    if temp_chunk:
                        chunk_text = " ".join(temp_chunk)
                        if len(chunk_text) >= 20:  # Basic length check
                            chunks.append({"text": chunk_text, "token_count": temp_length})
                    temp_chunk = [word]
                    temp_length += len(word_tokens)
                else:
                    temp_chunk.append(word)
                    temp_length += len(word_tokens)

            if temp_chunk:
                chunk_text = " ".join(temp_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({"text": chunk_text, "token_count": temp_length})
        # If adding this sentence would exceed chunk size, start new chunk
        elif current_length + sentence_length > n:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({"text": chunk_text, "token_count": current_length})
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add final chunk if exists
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= 20:  # Basic length check
            chunks.append({"text": chunk_text, "token_count": current_length})
    return chunks

async def retrieve_unique_strings(
    query: str,
    library_df: pd.DataFrame,
    agent: KnowledgeAgent,
    required_count: int = 10,  # Default to 5 chunks for better context
    provider: Optional[ModelProvider] = None,
) -> List[Dict[str, Any]]:
    """Retrieve unique strings from the library based on query relevance."""
    try:
        # Ensure we have appropriate text columns for searching
        if "text_clean" not in library_df.columns and "content" in library_df.columns:
            logger.info("Library using 'content' column (database source) for text search")
        elif "text_clean" in library_df.columns and "content" not in library_df.columns:
            logger.info("Library using 'text_clean' column (S3 source) for text search")
        elif "text_clean" in library_df.columns and "content" in library_df.columns:
            logger.info(
                "Library has both 'text_clean' and 'content' columns, will prioritize 'text_clean'"
            )
        else:
            logger.warning("Library missing both 'text_clean' and 'content' columns")

        # Get initial ranked strings
        ranked_strings = await strings_ranked_by_relatedness(
            query=query,
            df=library_df,
            agent=agent,
            top_n=required_count * 4,  # Get 4x more than needed to account for filtering
            provider=provider,
        )

        # Filter and deduplicate results
        seen_texts = set()
        unique_strings = []

        for item in ranked_strings:
            # Extract text from tuple result: (text, similarity, metadata)
            text = item[0]

            if text not in seen_texts and is_valid_chunk(text):
                seen_texts.add(text)
                unique_strings.append(item)
                if len(unique_strings) >= required_count:
                    break

        # Return at least one result if we have any
        if not unique_strings and ranked_strings:
            logger.warning(
                f"No valid chunks found among {len(ranked_strings)} results, returning first result"
            )
            unique_strings = [ranked_strings[0]]

        return unique_strings[:required_count]

    except Exception as e:
        logger.error(f"Error retrieving unique strings: {e}")
        traceback.print_exc()
        return []  # Return empty list on error instead of raising

async def process_query(
    query: str,
    agent: KnowledgeAgent,
    library_df: pd.DataFrame,
    config: Optional[Any] = None,
    provider_map: Optional[Dict[str, ModelProvider]] = None,
    use_batching: bool = True,
    character_slug: Optional[str] = None,
) -> Dict[str, Any]:
    """Process a query with configurable batching.

    Args:
        query: The user query string
        agent: KnowledgeAgent instance
        library_df: DataFrame containing stratified data with embeddings
        config: Optional configuration object
        provider_map: Optional mapping of operation types to providers
        use_batching: Whether to use batch processing (if False, processes one item at a time)
        character_slug: Optional character slug for generating chunks and summaries

    Returns:
        Dict containing chunks and summary
    """
    try:
        start_time = time.time()
        logger.info(f"Processing query: {query[:50]}... (batching: {use_batching})")
        
        # Set default providers if not specified
        if provider_map is None:
            provider_map = {
                ModelOperation.EMBEDDING: None,  # Use default from config
                ModelOperation.CHUNK_GENERATION: None,
                ModelOperation.SUMMARIZATION: None
            }
            
        # Get batch sizes from config or use defaults
        if config:
            # Get batch sizes from config with reasonable defaults
            embedding_batch_size = getattr(config, "embedding_batch_size", 25)
            chunk_batch_size = getattr(config, "chunk_batch_size", 25)
            summary_batch_size = getattr(config, "summary_batch_size", 25)

            logger.info(
                f"Using batch sizes from config - Embedding: {embedding_batch_size}, "
                + f"Chunk: {chunk_batch_size}, Summary: {summary_batch_size}"
            )
        else:
            # Default batch sizes if no config provided
            embedding_batch_size = 25
            chunk_batch_size = 25
            summary_batch_size = 25
            logger.info(
                f"No config provided, using default batch sizes - "
                + f"Embedding: {embedding_batch_size}, Chunk: {chunk_batch_size}, Summary: {summary_batch_size}"
            )
            
        # Step 1: Find relevant strings using embedding search
        strings = await retrieve_unique_strings(
            query=query,
            library_df=library_df,
            agent=agent,
            required_count=min(10, len(library_df)),
            provider=provider_map.get(ModelOperation.EMBEDDING)
        )
        
        # Skip processing if no relevant content
        if not strings:
            logger.warning("No relevant content found for query")
            return {"chunks": [], "summary": "No relevant content found."}
        
        logger.info(f"Found {len(strings)} relevant strings for query")
        
        # Step 2: Extract text content from strings for processing
        # Handle both tuple format (from strings_ranked_by_relatedness) and dictionary format
        texts = []
        for chunk in strings:
            if isinstance(chunk, tuple):
                # If it's a tuple (text, score, metadata), use the first element (text)
                texts.append(chunk[0])
            elif isinstance(chunk, dict) and "text_clean" in chunk:
                # If it's a dictionary with text_clean key, use that
                texts.append(chunk["text_clean"])
            else:
                logger.warning(f"Unexpected chunk format: {type(chunk)}")
                continue
        
        if not texts:
            logger.warning("No valid texts extracted from chunks")
            return {"chunks": [], "summary": "No valid texts found."}
        
        # Step 3: Process chunks (with or without batching)
        logger.info(f"Generating chunks for {len(texts)} texts using batch size {chunk_batch_size}")
        
        if use_batching:
            # Process using batching
            chunk_results = await agent.generate_chunks_batch(
                contents=texts,
                provider=provider_map.get(ModelOperation.CHUNK_GENERATION),
                chunk_batch_size=chunk_batch_size,
                character_slug=character_slug,
            )
        else:
            # Process one by one (for debugging or specific cases)
            chunk_results = []
            for content in texts:
                result = await agent.generate_chunks(
                    content=content,
                    provider=provider_map.get(ModelOperation.CHUNK_GENERATION),
                    character_slug=character_slug,
                )
                chunk_results.append(result)
        
        # Step 4: Prepare chunks with metadata for summary
        processed_chunks = []
        for string, result in zip(strings, chunk_results):
            if result:
                # Extract metadata based on the format of string
                if isinstance(string, tuple):
                    # Tuple format (text, score, metadata)
                    metadata = string[2]
                    processed_chunks.append(
                        {
                            "thread_id": metadata.get("thread_id", "unknown"),
                            "posted_date_time": metadata.get("posted_date_time", "unknown"),
                            "analysis": result,
                        }
                    )
                elif isinstance(string, dict):
                    # Dictionary format
                    processed_chunks.append(
                        {
                            "thread_id": string.get("thread_id", "unknown"),
                            "posted_date_time": string.get("posted_date_time", "unknown"),
                            "analysis": result,
                        }
                    )
        
        if not processed_chunks:
            logger.warning("No valid chunks generated from processing")
            return {"chunks": [], "summary": "No valid chunks generated."}
            
        logger.info(f"Generated {len(processed_chunks)} processed chunks")
        
        # --- NEW: Calculate temporal context from full library data for accurate range ---
        try:
            library_dates = pd.to_datetime(
                library_df["posted_date_time"], utc=True, errors="coerce"
            )
            valid_library_dates = library_dates[~pd.isna(library_dates)]
            if not valid_library_dates.empty:
                temporal_context = {
                    "start_date": valid_library_dates.min().strftime("%Y-%m-%d"),
                    "end_date": valid_library_dates.max().strftime("%Y-%m-%d"),
                }
            else:
                raise ValueError("No valid dates in library_df")
        except Exception as e:
            logger.warning(f"Falling back to processed chunks for temporal context due to: {e}")
            dates = pd.to_datetime(
                [r["posted_date_time"] for r in processed_chunks], utc=True, errors="coerce"
            )
            valid_dates = dates[~pd.isna(dates)]
            temporal_context = {
                "start_date": (
                    valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else "Unknown"
                ),
                "end_date": (
                    valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else "Unknown"
                ),
            }

        # Step 5: Generate summary (with or without batching)
        logger.info(
            f"Generating summary for {len(processed_chunks)} chunks with batch size {summary_batch_size}"
        )

        if use_batching:
            # Generate summary using batching
            summaries = await agent.generate_summaries_batch(
                queries=[query],
                results_list=[json.dumps(processed_chunks, indent=2)],
                contexts=[None],  # Add contexts parameter
                temporal_contexts=[temporal_context],
                provider=provider_map.get(ModelOperation.SUMMARIZATION),
                summary_batch_size=summary_batch_size,
                character_slug=character_slug,
            )
            summary = summaries[0] if summaries else "Failed to generate summary."
        else:
            # Generate summary without batching (one at a time)
            summary = await agent.generate_summary(
                query=query,
                results=json.dumps(processed_chunks, indent=2),
                temporal_context=temporal_context,
                provider=provider_map.get(ModelOperation.SUMMARIZATION),
                character_slug=character_slug,
            )

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Query processed in {duration_ms}ms (batching: {use_batching})")

        return {
            "chunks": processed_chunks,
            "summary": summary,
            "metadata": {
                "processing_time_ms": duration_ms,
                "num_relevant_strings": len(strings),
                "num_processed_chunks": len(processed_chunks),
                "temporal_context": temporal_context,
                "batch_sizes": {
                    "embedding": embedding_batch_size,
                    "chunk": chunk_batch_size,
                    "summary": summary_batch_size,
                },
                "batching_enabled": use_batching,
            },
        }
        
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        traceback.print_exc()
        return {"chunks": [], "summary": f"Error during processing: {str(e)}"}

async def process_multiple_queries_efficient(
    queries: List[str],
    agent: KnowledgeAgent,
    stratified_data: Optional[pd.DataFrame] = None,
    stratified_path: Optional[str] = None,
    chunk_batch_size: int = 10,
    summary_batch_size: int = 5,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None,
    character_slug: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process multiple queries efficiently using optimized batching.

    Args:
        queries: List of queries to process
        agent: KnowledgeAgent instance
        stratified_data: Optional pre-loaded DataFrame containing stratified data
        stratified_path: Optional path to stratified data directory (used if stratified_data not provided)
        chunk_batch_size: Batch size for chunk generation
        summary_batch_size: Batch size for summary generation
        max_workers: Maximum number of worker threads
        providers: Dictionary mapping operations to model providers
        character_slug: Optional character slug for generating chunks and summaries
        
    Returns:
        List of results for each query
    """
    try:
        start_time = time.time()
        logger.info(f"Processing {len(queries)} queries with efficient batching")
        
        if providers is None:
            providers = {}
            
        # Load stratified data if not provided
        library_df = stratified_data
        if library_df is None:
            if not stratified_path:
                logger.error("Neither stratified_data nor stratified_path was provided")
                raise ValueError("Either stratified_data or stratified_path must be provided")

            stratified_file = Path(stratified_path) / "stratified_sample.csv"
            embeddings_path = Path(stratified_path) / "embeddings.npz"
            thread_id_map_path = Path(stratified_path) / "thread_id_map.json"

            # Check if files exist before attempting to load
            if not stratified_file.exists():
                logger.error(f"Stratified file not found at {stratified_file}")
                raise FileNotFoundError(f"Stratified file not found at {stratified_file}")

            if not embeddings_path.exists():
                logger.error(f"Embeddings file not found at {embeddings_path}")
                raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}")

            if not thread_id_map_path.exists():
                logger.error(f"Thread ID map file not found at {thread_id_map_path}")
                raise FileNotFoundError(f"Thread ID map file not found at {thread_id_map_path}")

            try:
                # Load and merge stratified data with embeddings
                from .embedding_ops import merge_articles_and_embeddings

                library_df = await merge_articles_and_embeddings(
                    stratified_file, embeddings_path, thread_id_map_path
                )
                logger.info(f"Loaded stratified data with {len(library_df)} records")
            except Exception as e:
                logger.error(f"Error loading stratified data: {e}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Failed to load stratified data: {str(e)}")

        if library_df is None or library_df.empty:
            logger.error("Stratified dataset is empty or None")
            raise ValueError("Stratified dataset is empty or None")
            
        # Validate that the DataFrame has the required columns
        required_columns = ["text_clean", "thread_id", "posted_date_time"]
        missing_columns = [col for col in required_columns if col not in library_df.columns]
        if missing_columns:
            logger.error(f"Stratified data is missing required columns: {missing_columns}")
            raise ValueError(f"Stratified data is missing required columns: {missing_columns}")

        # Validate embeddings
        prepare_embeddings(library_df)
        embedding_count = library_df["embedding"].notna().sum()
        logger.info(
            f"Using stratified data with {len(library_df)} records and {embedding_count} embeddings"
        )

        # Create a mock config object with batch sizes
        class BatchConfig:
            def __init__(self, chunk_size, summary_size):
                self.chunk_batch_size = chunk_size
                self.summary_batch_size = summary_size
                
        config = BatchConfig(chunk_batch_size, summary_batch_size)
        
        # Process queries in parallel with controlled concurrency
        max_concurrent = min(max_workers or 5, len(queries))
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(query):
            async with semaphore:
                return await process_query(  # Updated to use process_query instead of process_query_efficient
                    query=query,
                    agent=agent,
                    library_df=library_df,
                    config=config,
                    provider_map={
                        ModelOperation.EMBEDDING: providers.get(ModelOperation.EMBEDDING),
                        ModelOperation.CHUNK_GENERATION: providers.get(
                            ModelOperation.CHUNK_GENERATION
                        ),
                        ModelOperation.SUMMARIZATION: providers.get(ModelOperation.SUMMARIZATION),
                    },
                    use_batching=True,  # Explicitly specify that we're using batching
                    character_slug=character_slug,
                )

        # Process all queries concurrently with controlled parallelism
        tasks = [process_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing query {i}: {str(result)}")
                processed_results.append(
                    {"chunks": [], "summary": f"Error: {str(result)}", "error": str(result)}
                )
            else:
                processed_results.append(result)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Processed {len(queries)} queries in {duration_ms}ms (avg: {duration_ms/len(queries):.2f}ms per query)"
        )
        
        return processed_results

    except Exception as e:
        logger.error(f"Error in process_multiple_queries_efficient: {e}")
        traceback.print_exc()
        # Return empty results for all queries
        return [{"chunks": [], "summary": f"Error: {str(e)}"} for _ in range(len(queries))]

async def get_query_matches(
    df: pd.DataFrame,
    query: str,
    agent: KnowledgeAgent,
    top_n: int = 10,
    provider: Optional[ModelProvider] = None,
) -> List[Dict[str, Any]]:
    """Returns a list of strings sorted from most related to least."""
    try:
        query_embedding_response = await agent.embedding_request(text=query, provider=provider)
        query_embedding = query_embedding_response.embedding

        embeddings, valid_indices, _ = prepare_embeddings(
            df=df,
            query_embedding=query_embedding,
            adapt=False,
        )

        # Calculate cosine similarities in one vectorized operation
        similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric="cosine")[0]

        # Get indices of top N similar items
        top_similarity_indices = np.argsort(similarities)[::-1][:top_n]

        # Map back to original dataframe indices
        top_df_indices = [valid_indices[i] for i in top_similarity_indices]

        # Return relevant records with similarity scores
        results = []
        for idx in top_df_indices:
            results.append(
                {
                    "text_clean": df.iloc[idx]["text_clean"],
                    "thread_id": df.iloc[idx]["thread_id"],
                    "similarity": float(
                        similarities[top_similarity_indices[top_df_indices.index(idx)]]
                    ),
                    "posted_date_time": df.iloc[idx]["posted_date_time"],
                }
            )

        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        traceback.print_exc()
        return []

async def recursive_strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    agent: KnowledgeAgent,
    final_top_n: int = 10,
    initial_top_n: int = 50,
    reduction_factor: float = 0.5,
    min_similarity_threshold: float = 0.5,
    max_iterations: int = 3,
    provider: Optional[ModelProvider] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Recursively refine content relatedness to a query for higher specificity.

    This function performs multiple passes of similarity ranking, progressively
    narrowing down the most relevant content by re-ranking at each step.

    Args:
        query: The query string to find related content for
        df: DataFrame containing text and embeddings
        agent: KnowledgeAgent instance for embedding generation
        final_top_n: Final number of results to return (default: 10)
        initial_top_n: Initial number of results to consider (default: 50)
        reduction_factor: Factor to reduce results by in each iteration (default: 0.5)
        min_similarity_threshold: Minimum similarity score to consider (default: 0.5)
        max_iterations: Maximum number of refinement iterations (default: 3)
        provider: Optional model provider to use for embeddings

    Returns:
        List of tuples containing (text, similarity_score, metadata)
    """
    # Validate inputs
    if df.empty:
        logger.warning("Empty dataframe provided to recursive_strings_ranked_by_relatedness")
        return []

    if final_top_n <= 0:
        logger.warning(f"Invalid final_top_n: {final_top_n}, using default of 10")
        final_top_n = 10

    # Initial retrieval with larger top_n
    current_top_n = initial_top_n
    current_results = await strings_ranked_by_relatedness(
        query=query, df=df, agent=agent, top_n=current_top_n, provider=provider
    )

    # If we don't have enough results or already at target size, return early
    if len(current_results) <= final_top_n:
        logger.info(
            f"Initial retrieval returned {len(current_results)} results, which is <= final_top_n ({final_top_n})"
        )
        return current_results

    # Track iterations for logging and limiting
    iteration = 1

    # Create a smaller dataframe with just the top results for refinement
    while len(current_results) > final_top_n and iteration < max_iterations:
        logger.info(
            f"Refinement iteration {iteration}: Refining from {len(current_results)} to {final_top_n} results"
        )

        # Extract texts and metadata from current results
        texts = [item[0] for item in current_results]
        scores = [item[1] for item in current_results]
        metadata = [item[2] for item in current_results]

        # Create a temporary dataframe with just these results
        temp_df = pd.DataFrame({"text": texts, "score": scores})

        # Add metadata columns if available
        for i, meta in enumerate(metadata):
            for key, value in meta.items():
                if key not in temp_df.columns:
                    temp_df[key] = None
                temp_df.at[i, key] = value

        # Re-embed these texts for more precise comparison
        # This is optional but can help with refinement quality
        temp_embeddings = await _get_embeddings_for_texts(texts, agent, provider)
        temp_df["embedding"] = temp_embeddings

        # Calculate next target size (using reduction factor)
        next_top_n = max(final_top_n, int(len(current_results) * reduction_factor))

        # Re-rank with the refined set
        current_results = await _rank_by_similarity(
            query=query, df=temp_df, agent=agent, top_n=next_top_n, provider=provider
        )

        iteration += 1

    # Final filtering by similarity threshold
    final_results = [result for result in current_results if result[1] >= min_similarity_threshold]

    logger.info(
        f"Recursive relatedness retrieval complete: {len(final_results)} results after {iteration} iterations"
    )
    return final_results[:final_top_n]


async def _get_embeddings_for_texts(
    texts: List[str], agent: KnowledgeAgent, provider: Optional[ModelProvider] = None
) -> List[np.ndarray]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        agent: KnowledgeAgent instance for embedding generation
        provider: Optional model provider to use

    Returns:
        List of embedding vectors
    """
    embeddings = []
    batch_size = 20  # Process in batches to avoid memory issues

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = await agent.get_embeddings(texts=batch, provider=provider)
        embeddings.extend(batch_embeddings)

    return embeddings


async def _rank_by_similarity(
    query: str,
    df: pd.DataFrame,
    agent: KnowledgeAgent,
    top_n: int = 10,
    provider: Optional[ModelProvider] = None,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Rank texts by similarity to query using cosine similarity.

    Args:
        query: Query string to compare against
        df: DataFrame with text and embedding columns
        agent: KnowledgeAgent for embedding generation
        top_n: Number of top results to return
        provider: Optional model provider to use

    Returns:
        List of (text, similarity_score, metadata) tuples
    """
    # Generate query embedding
    query_embedding = await agent.get_embeddings(texts=[query], provider=provider)

    if not query_embedding or len(query_embedding) == 0:
        logger.error("Failed to generate query embedding")
        return []

    # Extract embeddings from dataframe
    embeddings = np.array(df["embedding"].tolist())

    # Calculate similarities using cosine distance
    similarities = 1 - spatial.distance.cdist([query_embedding[0]], embeddings, metric="cosine")[0]

    # Get indices of top results
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Build result tuples with metadata
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        text = row["text"]
        score = float(similarities[idx])

        # Extract metadata (all columns except text and embedding)
        metadata = {}
        for col in df.columns:
            if col not in ["text", "embedding"]:
                metadata[col] = row[col]

        results.append((text, score, metadata))

    return results