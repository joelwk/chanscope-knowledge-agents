import ast
import json
import pandas as pd
from scipy import spatial
import tiktoken
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from .model_ops import KnowledgeAgent, ModelProvider, ModelOperation
from .embedding_ops import get_relevant_content
import logging
import numpy as np

# Initialize logging
logger = logging.getLogger(__name__)

def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    agent: KnowledgeAgent,
    top_n: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[str]:
    """Returns a list of strings sorted from most related to least."""
    try:
        query_embedding_response = agent.embedding_request(
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
        return df.iloc[top_indices][["text", "thread_id", "posted_date_time"]].to_dict('records')
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

def retrieve_unique_strings(
    query: str,
    library_df: pd.DataFrame,
    agent: KnowledgeAgent,
    required_count: int = 100,
    provider: Optional[ModelProvider] = None
) -> List[Dict[str, Any]]:
    """Fetches a specified number of unique top strings."""
    try:
        strings = strings_ranked_by_relatedness(
            query,
            library_df,
            agent,
            top_n=required_count * 2,  # Get more initially to account for duplicates
            provider=provider)
        
        # Use dict to maintain order while removing duplicates
        seen = set()
        unique_strings = []
        for s in strings:
            if s["thread_id"] not in seen:
                seen.add(s["thread_id"])
                unique_strings.append(s)
                if len(unique_strings) >= required_count:
                    break
                    
        if len(unique_strings) < required_count:
            logger.info(f"Only {len(unique_strings)} unique strings found, retrieving additional...")
            more_strings = strings_ranked_by_relatedness(
                query,
                library_df,
                agent,
                top_n=required_count * 4,  # Try with even more
                provider=provider)
            
            for s in more_strings:
                if s["thread_id"] not in seen:
                    seen.add(s["thread_id"])
                    unique_strings.append(s)
                    if len(unique_strings) >= required_count:
                        break
                        
        return unique_strings[:required_count]
        
    except Exception as e:
        logger.error(f"Error retrieving unique strings: {e}")
        raise

async def summarize_text(
    query: str,
    agent: KnowledgeAgent,
    knowledge_base_path: str = ".",
    batch_size: int = 5,
    max_workers: Optional[int] = None,
    providers: Optional[Dict[ModelOperation, ModelProvider]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Summarizes related texts from the knowledge base with temporal forecasting."""
    try:
        # Ensure we have providers
        if providers is None:
            providers = {
                ModelOperation.EMBEDDING: ModelProvider.OPENAI,
                ModelOperation.CHUNK_GENERATION: ModelProvider.GROK,
                ModelOperation.SUMMARIZATION: ModelProvider.VENICE
            }
            
        # Initialize clients for the providers we'll use
        agent._get_client(providers[ModelOperation.EMBEDDING])
        agent._get_client(providers[ModelOperation.CHUNK_GENERATION])
        agent._get_client(providers[ModelOperation.SUMMARIZATION])
        
        # Load and prepare knowledge base
        library_df = pd.read_csv(knowledge_base_path)
        if len(library_df) == 0:
            logger.info("No data found, downloading first.")
            get_relevant_content(query, batch_size=batch_size)
            library_df = pd.read_csv(knowledge_base_path)
        
        library_df.columns = ["thread_id", "posted_date_time", "text", "embedding"]
        library_df['posted_date_time'] = pd.to_datetime(library_df['posted_date_time'])
        
        # Filter out rows with very short text
        library_df = library_df[library_df['text'].str.len() >= 20]
        
        # Get unique related strings
        embedding_provider = providers.get(ModelOperation.EMBEDDING)
        unique_strings = retrieve_unique_strings(
            query,
            library_df,
            agent,
            provider=embedding_provider
        )
        
        logger.info("Processing text chunks")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        all_chunks = []
        
        # Process each text into chunks
        for item in unique_strings:
            if len(item['text'].strip()) < 20:  # Skip very short texts
                continue
                
            chunk_text = f"""<temporal_context>
                Posted: {item['posted_date_time']}
                Thread: {item['thread_id']}
                </temporal_context>
                <content>
                {item['text']}
                </content>"""
            
            # Clean and validate chunk text
            chunk_text = clean_chunk_text(chunk_text)
            chunks = create_chunks(chunk_text, 1000, tokenizer)
            
            # Add only valid chunks
            for chunk in chunks:
                if is_valid_chunk(chunk["text"]):
                    all_chunks.append(chunk)
        
        if not all_chunks:
            logger.warning("No valid chunks found to process")
            return [], ""
        
        logger.info(f"Summarizing {len(all_chunks)} valid chunks of text")
        
        # Configure worker pool
        if max_workers is None:
            max_workers = min(8, len(all_chunks))
        
        summaries = []
        contexts = []
        chunk_provider = providers.get(ModelOperation.CHUNK_GENERATION) if providers else None
        
        # Process chunks directly
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            # Submit all chunks for processing
            for chunk in all_chunks:
                future = executor.submit(
                    agent.generate_chunks,
                    chunk["text"],
                    provider=chunk_provider
                )
                futures.append(future)
            
            # Collect results with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                try:
                    result = future.result()
                    if result["analysis"].strip():
                        summaries.append(result["analysis"])
                    if result["context"].strip():
                        contexts.append(result["context"])
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}")
                    continue
        
        if not summaries:
            logger.warning("No summaries generated")
            return all_chunks, ""
        
        # Combine summaries and contexts
        combined_summary = "\n\n".join(summaries)
        combined_context = "\n\n".join(contexts)
        
        try:
            logger.info("Generating final summary")
            summary_provider = providers.get(ModelOperation.SUMMARIZATION) if providers else None
            
            # Get date range from unique strings
            dates = pd.to_datetime([s['posted_date_time'] for s in unique_strings])
            start_date = min(dates)
            end_date = max(dates)
            
            # Format temporal context to match the expected template variables
            temporal_context = {
                "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
                "end_date": end_date.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Format the combined summaries into structured sections
            formatted_results = []
            event_map = {}  # Map to store event descriptions
            event_counter = 1
            
            for chunk in summaries:
                if chunk.strip():
                    # Extract metrics and analysis from chunk summary
                    sections = chunk.split("<signal_context>")
                    if len(sections) > 1:
                        metrics = sections[0].strip()
                        analysis = sections[1].strip()
                        
                        # Extract key claims and topics for event naming
                        claims_section = analysis.split("Key claims detected:")
                        if len(claims_section) > 1:
                            claim_text = claims_section[1].split("\n")[0].strip()
                            # Clean and shorten claim for event name
                            event_name = claim_text.split(".")[0].strip()[:50]  # Take first sentence, limit length
                            event_name = event_name.replace('"', '').replace("'", "")
                            event_map[f"event{event_counter}"] = event_name
                            event_counter += 1
                        
                        formatted_results.append(f"<analysis_section>\n{metrics}\n{analysis}\n</analysis_section>")
            
            structured_results = "\n\n".join(formatted_results)
            
            # Add event mapping to the context
            context_with_events = f"""
            {combined_context}
            
            <event_mapping>
            {json.dumps(event_map, indent=2)}
            </event_mapping>
            """
            
            response = agent.generate_summary(
                query, 
                structured_results,
                context=context_with_events,
                temporal_context=temporal_context,
                provider=summary_provider
            )
            
            if not response or response.strip() == "":
                logger.warning("Empty summary response, falling back to structured format")
                # Create a basic structured summary
                response = f"""### 1. Temporal Overview
Time Range: {temporal_context['start_date']} to {temporal_context['end_date']}
Active Periods: Multiple threads analyzed across the time range
Thread Distribution: See detailed metrics below

### 2. Thread Analysis
{structured_results}"""
            
            return all_chunks, response
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")
            # Create a structured fallback summary
            fallback = f"""### 1. Temporal Overview
Time Range: {start_date.strftime("%Y-%m-%d %H:%M:%S")} to {end_date.strftime("%Y-%m-%d %H:%M:%S")}
Active Periods: Multiple threads analyzed across the time range
Thread Distribution: See detailed metrics below

### 2. Thread Analysis
{combined_summary[:3000]}"""  # Limit length of fallback
            return all_chunks, fallback
            
    except Exception as e:
        logger.error(f"Error in summarization pipeline: {e}")
        raise