import ast
import json
import pandas as pd
from scipy import spatial
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from .model_ops import KnowledgeAgent, ModelProvider, ModelOperation
import logging
import numpy as np
from pathlib import Path
import asyncio
import traceback

# Initialize logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

async def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    agent: KnowledgeAgent,
    top_n: int = 50,
    provider: Optional[ModelProvider] = None
) -> List[Dict[str, Any]]:
    """Returns a list of strings sorted from most related to least."""
    try:
        query_embedding_response = await agent.embedding_request(
            text=query,
            provider=provider)
        query_embedding = query_embedding_response.embedding
        
        # Handle embeddings based on their format
        if 'embedding' not in df.columns:
            logger.error("No embeddings found in DataFrame")
            raise ValueError("DataFrame must contain 'embedding' column")
            
        # Convert embeddings to numpy array, handling different formats
        embeddings = []
        valid_indices = []  # Keep track of valid indices to map back to original dataframe
        
        for i, emb in enumerate(df['embedding'].values):
            if isinstance(emb, str):
                # Handle JSON string format
                try:
                    parsed_emb = json.loads(emb)
                    if parsed_emb:  # Ensure it's not empty
                        embeddings.append(parsed_emb)
                        valid_indices.append(i)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse embedding JSON: {emb[:100]}...")
                    continue
            elif isinstance(emb, list):
                if not emb:  # Skip empty lists
                    continue
                embeddings.append(emb)
                valid_indices.append(i)
            elif isinstance(emb, np.ndarray):
                if emb.size == 0:  # Skip empty arrays
                    continue
                elif emb.size == 1:  # Handle single-element arrays
                    # Create a dummy embedding
                    logger.warning(f"Converting single-item array at index {i} to vector")
                    embeddings.append([float(emb.item()), 0.0, 0.0])
                    valid_indices.append(i)
                else:
                    embeddings.append(emb)
                    valid_indices.append(i)
            elif isinstance(emb, (float, int)):
                # Handle single float values
                logger.warning(f"Converting single float value at index {i} to array")
                embeddings.append([float(emb), 0.0, 0.0])
                valid_indices.append(i)
            else:
                logger.warning(f"Unexpected embedding format at index {i}: {type(emb)}")
                continue
                
        if not embeddings:
            raise ValueError("No valid embeddings found in DataFrame")
        
        # Check if all embeddings have the same dimension
        embedding_dimensions = [len(e) if hasattr(e, '__len__') else 1 for e in embeddings]
        if len(set(embedding_dimensions)) > 1:
            logger.warning(f"Mixed embedding dimensions found: {set(embedding_dimensions)}")
            # Filter to keep only embeddings with the same dimension as the query
            query_dim = len(query_embedding) if hasattr(query_embedding, '__len__') else 1
            filtered_embeddings = []
            filtered_indices = []
            for i, (emb, orig_idx) in enumerate(zip(embeddings, valid_indices)):
                emb_dim = len(emb) if hasattr(emb, '__len__') else 1
                if emb_dim == query_dim:
                    filtered_embeddings.append(emb)
                    filtered_indices.append(orig_idx)
            
            if not filtered_embeddings:
                # If no embeddings match the query dimension, use dimension adaptation
                logger.warning(f"No embeddings with matching dimension {query_dim} found, using dimension adaptation")
                
                # First check if we have any reasonably sized embeddings (beyond our simple 3-element placeholders)
                substantial_embeddings = []
                substantial_indices = []
                
                for i, (emb, orig_idx) in enumerate(zip(embeddings, valid_indices)):
                    if hasattr(emb, '__len__') and len(emb) > 3:
                        substantial_embeddings.append(emb)
                        substantial_indices.append(orig_idx)
                
                if substantial_embeddings:
                    logger.info(f"Using {len(substantial_embeddings)} substantial embeddings with dimension adaptation")
                    embeddings = substantial_embeddings
                    valid_indices = substantial_indices
                    
                    # Convert to numpy array 
                    embeddings = np.vstack(embeddings)
                    
                    # Use only the first min(len(embeddings[0]), len(query_embedding)) dimensions for comparison
                    min_dim = min(embeddings.shape[1], len(query_embedding))
                    
                    # Truncate both embeddings and query to the minimum dimension
                    truncated_query = query_embedding[:min_dim]
                    truncated_embeddings = embeddings[:, :min_dim]
                    
                    # Calculate similarities using truncated dimensions
                    similarities = 1 - spatial.distance.cdist([truncated_query], truncated_embeddings, metric='cosine')[0]
                else:
                    # If no substantial embeddings, fall back to all embeddings and pad as needed
                    logger.warning("Falling back to simple dimension padding for embeddings")
                    
                    # Pad short embeddings or truncate long ones to match query_dim
                    processed_embeddings = []
                    for emb in embeddings:
                        if hasattr(emb, '__len__'):
                            current_len = len(emb)
                            if current_len < query_dim:
                                # Pad with zeros
                                padded = np.zeros(query_dim, dtype=np.float32)
                                padded[:current_len] = emb
                                processed_embeddings.append(padded)
                            else:
                                # Truncate
                                processed_embeddings.append(emb[:query_dim])
                        else:
                            # Single value - expand to array
                            padded = np.zeros(query_dim, dtype=np.float32)
                            padded[0] = float(emb)
                            processed_embeddings.append(padded)
                    
                    # Convert to numpy array after processing
                    embeddings = np.vstack(processed_embeddings)
                    
                    # Calculate similarities
                    similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
            else:
                embeddings = filtered_embeddings
                valid_indices = filtered_indices
                logger.info(f"Filtered to {len(embeddings)} embeddings with dimension {query_dim}")
                
                # Convert to numpy array for efficient computation
                try:
                    embeddings = np.vstack(embeddings)
                except ValueError as e:
                    logger.error(f"Failed to stack embeddings: {e}")
                    # Try to convert any remaining problematic embeddings
                    embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
                    embeddings = np.vstack(embeddings)
                
                # Calculate cosine similarities in one vectorized operation
                similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
        else:
            # All embeddings have the same dimension, proceed normally
            try:
                embeddings = np.vstack(embeddings)
            except ValueError as e:
                logger.error(f"Failed to stack embeddings: {e}")
                # Try to convert any remaining problematic embeddings
                embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
                embeddings = np.vstack(embeddings)
            
            # Calculate cosine similarities in one vectorized operation
            similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
        
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
            results.append({
                "text_clean": df.iloc[idx]["text_clean"],
                "thread_id": str(df.iloc[idx]["thread_id"]),
                "posted_date_time": str(df.iloc[idx]["posted_date_time"]),
                "similarity_score": float(sim_score)
            })
            
        return results
        
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
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line and not line.isspace():
            if any(tag in line for tag in ['<temporal_context>', '</temporal_context>', '<content>', '</content>']):
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.strip())

    return '\n'.join(cleaned_lines)

def create_chunks(text: str, n: int, tokenizer=None) -> List[Dict[str, Any]]:
    """Creates chunks of text, preserving sentence boundaries where possible."""
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # First, split text into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        # Add period back to sentence
        sentence = sentence + '.'
        sentence_tokens = tokenizer.encode(sentence)
        sentence_length = len(sentence_tokens)

        # If single sentence is too long, split it
        if sentence_length > n:
            # If we have a current chunk, add it first
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({
                        "text": chunk_text,
                        "token_count": current_length
                    })
                current_chunk = []
                current_length = 0

            # Split long sentence into chunks
            words = sentence.split()
            temp_chunk = []
            temp_length = 0

            for word in words:
                word_tokens = tokenizer.encode(word + ' ')
                if temp_length + len(word_tokens) > n:
                    if temp_chunk:
                        chunk_text = ' '.join(temp_chunk)
                        if len(chunk_text) >= 20:  # Basic length check
                            chunks.append({
                                "text": chunk_text,
                                "token_count": temp_length
                            })
                    temp_chunk = [word]
                    temp_length += len(word_tokens)
                else:
                    temp_chunk.append(word)
                    temp_length += len(word_tokens)

            if temp_chunk:
                chunk_text = ' '.join(temp_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({
                        "text": chunk_text,
                        "token_count": temp_length
                    })
        # If adding this sentence would exceed chunk size, start new chunk
        elif current_length + sentence_length > n:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= 20:  # Basic length check
                    chunks.append({
                        "text": chunk_text,
                        "token_count": current_length
                    })
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add final chunk if exists
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= 20:  # Basic length check
            chunks.append({
                "text": chunk_text,
                "token_count": current_length
            })
    return chunks

async def retrieve_unique_strings(
    query: str,
    library_df: pd.DataFrame,
    agent: KnowledgeAgent,
    required_count: int = 5,  # Default to 5 chunks for better context
    provider: Optional[ModelProvider] = None
) -> List[Dict[str, Any]]:
    """Retrieve unique strings from the library based on query relevance."""
    try:
        # Get initial ranked strings
        ranked_strings = await strings_ranked_by_relatedness(
            query=query,
            df=library_df,
            agent=agent,
            top_n=required_count * 4,  # Get 4x more than needed to account for filtering
            provider=provider
        )

        # Filter and deduplicate results
        seen_texts = set()
        unique_strings = []

        for item in ranked_strings:
            # Extract text from dictionary result
            text = item["text_clean"]
            
            if text not in seen_texts and is_valid_chunk(text):
                seen_texts.add(text)
                unique_strings.append(item)
                if len(unique_strings) >= required_count:
                    break

        # Return at least one result if we have any
        if not unique_strings and ranked_strings:
            logger.warning(f"No valid chunks found among {len(ranked_strings)} results, returning first result")
            unique_strings = [ranked_strings[0]]
            
        return unique_strings[:required_count]

    except Exception as e:
        logger.error(f"Error retrieving unique strings: {e}")
        traceback.print_exc()
        return []  # Return empty list on error instead of raising

async def process_multiple_queries(
    queries: List[str],
    agent: KnowledgeAgent,
    stratified_data: Optional[pd.DataFrame] = None,
    stratified_path: Optional[str] = None,
    chunk_batch_size: int = 20,
    summary_batch_size: int = 20,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None
) -> List[Tuple[List[Dict[str, Any]], str]]:
    """Process multiple queries in parallel using asyncio for improved performance.

    Args:
        queries: List of queries to process
        agent: KnowledgeAgent instance
        stratified_data: Optional pre-loaded DataFrame containing stratified data
        stratified_path: Optional path to stratified data directory (used if stratified_data not provided)
        chunk_batch_size: Batch size for chunk generation
        summary_batch_size: Batch size for summary generation
        max_workers: Maximum number of worker threads
        providers: Dictionary mapping operations to model providers
    """
    try:
        if providers is None:
            providers = {}
            
        # Load stratified data if not provided
        library_df = stratified_data
        if library_df is None:
            if not stratified_path:
                raise ValueError("Either stratified_data or stratified_path must be provided")
            
            stratified_file = Path(stratified_path) / 'stratified_sample.csv'
            embeddings_path = Path(stratified_path) / 'embeddings.npz'
            thread_id_map_path = Path(stratified_path) / 'thread_id_map.json'
            
            try:
                # Load and merge stratified data with embeddings
                from .embedding_ops import merge_articles_and_embeddings
                library_df = await merge_articles_and_embeddings(
                    stratified_file,
                    embeddings_path,
                    thread_id_map_path
                )
            except Exception as e:
                logger.error(f"Error loading stratified data: {e}")
                raise ValueError("Failed to load stratified data")

        if library_df.empty:
            raise ValueError("Stratified dataset is empty")

        # Extract providers
        embedding_provider = providers.get(ModelOperation.EMBEDDING)
        chunk_provider = providers.get(ModelOperation.CHUNK_GENERATION)
        summary_provider = providers.get(ModelOperation.SUMMARIZATION)

        # Define async function to process a single query
        async def process_single_query(query):
            try:
                # Get relevant strings for this query
                strings = await retrieve_unique_strings(
                    query,
                    library_df,
                    agent,
                    required_count=min(50, len(library_df)),
                    provider=embedding_provider
                )
                
                if not strings:
                    return ([], "No relevant content found.")
                    
                # Get texts for chunk generation
                texts = [chunk["text_clean"] for chunk in strings]
                
                # Generate chunks for this query
                chunk_results_raw = await agent.generate_chunks_batch(
                    contents=texts,
                    provider=chunk_provider,
                    chunk_batch_size=chunk_batch_size
                )
                
                # Combine results with metadata
                chunk_results = []
                for chunk, result in zip(strings, chunk_results_raw):
                    if result:
                        chunk_results.append({
                            "thread_id": chunk["thread_id"],
                            "posted_date_time": chunk["posted_date_time"],
                            "analysis": result
                        })
                        
                # If no valid chunks were generated, return empty results
                if not chunk_results:
                    return ([], "No valid chunks generated.")
                    
                # Prepare for summary generation
                dates = pd.to_datetime([r["posted_date_time"] for r in chunk_results], utc=True, errors='coerce')
                
                # Handle missing or invalid dates
                valid_dates = dates[~pd.isna(dates)]
                if len(valid_dates) == 0:
                    temporal_context = {
                        "start_date": "Unknown",
                        "end_date": "Unknown"
                    }
                else:
                    temporal_context = {
                        "start_date": valid_dates.min().strftime("%Y-%m-%d"),
                        "end_date": valid_dates.max().strftime("%Y-%m-%d")
                    }
                
                # Generate summary
                summary_input = json.dumps(chunk_results, indent=2)
                summary = await agent.generate_summary(
                    query=query,
                    results=summary_input,
                    temporal_context=temporal_context,
                    provider=summary_provider
                )
                
                return (chunk_results, summary or "Failed to generate summary.")
            except Exception as e:
                logger.error(f"Error processing single query: {e}")
                traceback.print_exc()
                return ([], f"Error during processing: {str(e)}")
        
        # Process all queries concurrently with controlled concurrency
        max_concurrent = min(10, len(queries))  # Limit concurrency to 10 or fewer
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(query):
            async with semaphore:
                return await process_single_query(query)
        
        tasks = [process_with_semaphore(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing query {i}: {str(result)}")
                processed_results.append(([], f"Error: {str(result)}"))
            else:
                processed_results.append(result)
        
        return processed_results

    except Exception as e:
        logger.error(f"Error in process_multiple_queries: {e}")
        traceback.print_exc()
        # Return empty results for all queries
        return [([], f"Error: {str(e)}") for _ in range(len(queries))]

async def get_query_matches(df: pd.DataFrame, query: str, agent: KnowledgeAgent, 
                         top_n: int = 10, 
                         provider: Optional[ModelProvider] = None) -> List[Dict[str, Any]]:
    """Returns a list of strings sorted from most related to least."""
    try:
        query_embedding_response = await agent.embedding_request(
            text=query,
            provider=provider)
        query_embedding = query_embedding_response.embedding
        
        # Handle embeddings based on their format
        if 'embedding' not in df.columns:
            logger.error("No embeddings found in DataFrame")
            raise ValueError("DataFrame must contain 'embedding' column")
            
        # Convert embeddings to numpy array, handling different formats
        embeddings = []
        valid_indices = []  # Keep track of valid indices to map back to original dataframe
        
        for i, emb in enumerate(df['embedding'].values):
            if isinstance(emb, str):
                # Handle JSON string format
                try:
                    parsed_emb = json.loads(emb)
                    if parsed_emb:  # Ensure it's not empty
                        embeddings.append(parsed_emb)
                        valid_indices.append(i)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse embedding JSON: {emb[:100]}...")
                    continue
            elif isinstance(emb, list):
                if not emb:  # Skip empty lists
                    continue
                embeddings.append(emb)
                valid_indices.append(i)
            elif isinstance(emb, np.ndarray):
                if emb.size == 0:  # Skip empty arrays
                    continue
                elif emb.size == 1:  # Handle single-element arrays
                    # Create a dummy embedding
                    logger.warning(f"Converting single-item array at index {i} to vector")
                    embeddings.append([float(emb.item()), 0.0, 0.0])
                    valid_indices.append(i)
                else:
                    embeddings.append(emb)
                    valid_indices.append(i)
            elif isinstance(emb, (float, int)):
                # Handle single float values
                logger.warning(f"Converting single float value at index {i} to array")
                embeddings.append([float(emb), 0.0, 0.0])
                valid_indices.append(i)
            else:
                logger.warning(f"Unexpected embedding format at index {i}: {type(emb)}")
                continue
                
        if not embeddings:
            raise ValueError("No valid embeddings found in DataFrame")
        
        # Check if all embeddings have the same dimension
        embedding_dimensions = [len(e) if hasattr(e, '__len__') else 1 for e in embeddings]
        if len(set(embedding_dimensions)) > 1:
            logger.warning(f"Mixed embedding dimensions found: {set(embedding_dimensions)}")
            # Filter to keep only embeddings with the same dimension as the query
            query_dim = len(query_embedding) if hasattr(query_embedding, '__len__') else 1
            filtered_embeddings = []
            filtered_indices = []
            for i, (emb, orig_idx) in enumerate(zip(embeddings, valid_indices)):
                emb_dim = len(emb) if hasattr(emb, '__len__') else 1
                if emb_dim == query_dim:
                    filtered_embeddings.append(emb)
                    filtered_indices.append(orig_idx)
            
            if not filtered_embeddings:
                raise ValueError(f"No embeddings with matching dimension {query_dim} found")
            
            embeddings = filtered_embeddings
            valid_indices = filtered_indices
            logger.info(f"Filtered to {len(embeddings)} embeddings with dimension {query_dim}")
            
        # Convert to numpy array for efficient computation
        try:
            embeddings = np.vstack(embeddings)
        except ValueError as e:
            logger.error(f"Failed to stack embeddings: {e}")
            # Try to convert any remaining problematic embeddings
            embeddings = [np.array(e, dtype=np.float32) for e in embeddings]
            embeddings = np.vstack(embeddings)
        
        # Calculate cosine similarities in one vectorized operation
        similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
        
        # Get indices of top N similar items
        top_similarity_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Map back to original dataframe indices
        top_df_indices = [valid_indices[i] for i in top_similarity_indices]
        
        # Return relevant records with similarity scores
        results = []
        for idx in top_df_indices:
            results.append({
                "text_clean": df.iloc[idx]["text_clean"],
                "thread_id": df.iloc[idx]["thread_id"],
                "similarity": float(similarities[top_similarity_indices[top_df_indices.index(idx)]]),
                "posted_date_time": df.iloc[idx]["posted_date_time"],
            })
            
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        traceback.print_exc()
        return []