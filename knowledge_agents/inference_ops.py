import ast
import json
import pandas as pd
from scipy import spatial
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from .model_ops import KnowledgeAgent, ModelProvider, ModelOperation
import logging
import numpy as np

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
) -> List[str]:
    """Returns a list of strings sorted from most related to least."""
    try:
        query_embedding_response = await agent.embedding_request(
            text=query,
            provider=provider)
        query_embedding = query_embedding_response.embedding
        # Convert string embeddings back to lists if they're stored as strings
        if isinstance(df["embedding"].iloc[0], str):
            df["embedding"] = df["embedding"].apply(ast.literal_eval)
        # Convert embeddings to numpy array for vectorized operations
        embeddings = np.vstack(df["embedding"].values)
        # Calculate cosine similarities in one vectorized operation
        similarities = 1 - spatial.distance.cdist([query_embedding], embeddings, metric='cosine')[0]
        # Get indices of top N similar items
        top_indices = np.argsort(similarities)[::-1][:top_n]
        return df.iloc[top_indices][["text_clean", "thread_id", "posted_date_time"]].to_dict('records')
    except Exception as e:
        logger.error(f"Error ranking strings by relatedness: {e}")
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
                    temp_length = len(word_tokens)
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
    required_count: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[Dict[str, Any]]:
    """Retrieve unique strings from the library based on query relevance."""
    try:
        # Get initial ranked strings
        ranked_strings = await strings_ranked_by_relatedness(
            query=query,
            df=library_df,
            agent=agent,
            top_n=required_count * 2,  # Get more than needed to account for filtering
            provider=provider
        )

        # Filter and deduplicate results
        seen_texts = set()
        unique_strings = []

        for item in ranked_strings:
            text = item["text_clean"]
            if text not in seen_texts and is_valid_chunk(text):
                seen_texts.add(text)
                unique_strings.append(item)
                if len(unique_strings) >= required_count:
                    break

        return unique_strings[:required_count]

    except Exception as e:
        logger.error(f"Error retrieving unique strings: {e}")
        raise

async def process_multiple_queries(
    queries: List[str],
    agent: KnowledgeAgent,
    knowledge_base_path: str = ".",
    chunk_batch_size: int = 20,  # OpenAI recommended batch size for chunks
    summary_batch_size: int = 20,  # OpenAI recommended batch size for summaries
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None
) -> List[Tuple[List[Dict[str, Any]], str]]:
    """Process multiple queries in parallel using batch processing.

    Args:
        queries: List of queries to process
        agent: KnowledgeAgent instance
        knowledge_base_path: Path to knowledge base
        chunk_batch_size: Batch size for chunk generation (OpenAI recommends 20)
        summary_batch_size: Batch size for summary generation (OpenAI recommends 20)
        max_workers: Maximum number of worker threads
        providers: Dictionary mapping operations to model providers
    """
    try:
        # Load knowledge base once for all queries
        try:
            library_df = pd.read_csv(knowledge_base_path)
        except Exception as e:
            logger.error(f"Error reading knowledge base: {e}")
            raise ValueError("Failed to read knowledge base")

        if library_df.empty:
            raise ValueError("Knowledge base is empty")

        # Process all queries in parallel
        embedding_provider = providers.get(ModelOperation.EMBEDDING)
        chunk_provider = providers.get(ModelOperation.CHUNK_GENERATION)
        summary_provider = providers.get(ModelOperation.SUMMARIZATION)

        # Get relevant strings for all queries
        all_strings = []
        for query in queries:
            strings = await retrieve_unique_strings(
                query,
                library_df,
                agent,
                required_count=chunk_batch_size,  # Use chunk batch size for retrieval
                provider=embedding_provider
            )
            all_strings.append(strings)

        # Process chunks in batches for all queries
        all_chunk_results = []
        for strings in all_strings:
            if not strings:
                all_chunk_results.append([])
                continue

            texts = [chunk["text_clean"] for chunk in strings]
            chunk_results_raw = await agent.generate_chunks_batch(
                contents=texts,
                provider=chunk_provider,
                chunk_batch_size=chunk_batch_size  # Use chunk-specific batch size
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
            all_chunk_results.append(chunk_results)

        # Generate summaries in batches
        summary_inputs = []
        temporal_contexts = []
        for chunk_results in all_chunk_results:
            if not chunk_results:
                summary_inputs.append("")
                temporal_contexts.append({
                    "start_date": "Unknown",
                    "end_date": "Unknown"
                })
                continue

            dates = pd.to_datetime([r["posted_date_time"] for r in chunk_results], utc=True)
            temporal_contexts.append({
                "start_date": dates.min().strftime("%Y-%m-%d"),
                "end_date": dates.max().strftime("%Y-%m-%d")
            })
            summary_inputs.append(json.dumps(chunk_results, indent=2))

        # Generate summaries in batches
        summaries = await agent.generate_summaries_batch(
            queries=queries,
            results_list=summary_inputs,
            temporal_contexts=temporal_contexts,
            provider=summary_provider,
            summary_batch_size=summary_batch_size  # Use summary-specific batch size
        )

        # Combine results
        results = []
        for chunk_results, summary in zip(all_chunk_results, summaries):
            results.append((chunk_results or [], summary or "Failed to generate summary."))

        return results

    except Exception as e:
        logger.error(f"Error in process_multiple_queries: {e}")
        raise