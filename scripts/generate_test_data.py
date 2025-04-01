#!/usr/bin/env python3
"""
Test Data Generator for Chanscope

This script generates synthetic data for testing the Chanscope system when
real data is unavailable or needs to be augmented with more recent timestamps.
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
import numpy as np
import datetime
from pathlib import Path
import logging
import random

# Ensure python path includes the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import after path setup
from config.chanscope_config import ChanScopeConfig
from knowledge_agents.data_processing.chanscope_manager import ChanScopeDataManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default column names based on chanscope schema
COLUMNS = [
    'thread_id', 'post_id', 'post_text', 'posted_date_time', 'reply_to', 
    'subject', 'author', 'board_id', 'country_code', 'post_url', 'images'
]

def generate_board_id():
    """Generate a realistic board ID."""
    boards = ['pol', 'biz', 'sci', 'g', 'v', 'x', 'fit', 'news']
    return random.choice(boards)

def generate_country_code():
    """Generate a realistic country code."""
    countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'RU', 'BR', None]
    return random.choice(countries)

def generate_text(min_length=10, max_length=200):
    """Generate random post text."""
    lorem_ipsum = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
    irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
    deserunt mollit anim id est laborum.
    """
    
    words = lorem_ipsum.strip().split()
    length = random.randint(min_length, max_length)
    # Sometimes add special characters to simulate chan style posts
    specialchars = ['>', '>>1234', '(((them)))', '*laughs*', '[citation needed]', '&gt;', 'kek']
    
    if random.random() < 0.3:
        words.insert(0, random.choice(specialchars))
    
    return ' '.join(random.sample(words, min(length, len(words))))

def generate_synthetic_data(num_rows=1000, start_date=None, end_date=None):
    """
    Generate synthetic data for testing.
    
    Args:
        num_rows: Number of rows to generate
        start_date: Start date for timestamps (defaults to 10 days ago)
        end_date: End date for timestamps (defaults to now)
    
    Returns:
        DataFrame with synthetic data
    """
    if not start_date:
        start_date = datetime.datetime.now() - datetime.timedelta(days=10)
    if not end_date:
        end_date = datetime.datetime.now()
    
    # Generate timestamps distributed between start and end dates
    date_range = (end_date - start_date).total_seconds()
    timestamps = [
        start_date + datetime.timedelta(seconds=random.randint(0, date_range))
        for _ in range(num_rows)
    ]
    timestamps.sort()  # Sort chronologically
    
    # Generate thread IDs (some posts should belong to the same thread)
    num_threads = max(1, num_rows // 10)  # Average 10 posts per thread
    thread_ids = [f"{random.randint(10000000, 99999999)}" for _ in range(num_threads)]
    
    # Assign threads to posts
    assigned_thread_ids = [random.choice(thread_ids) for _ in range(num_rows)]
    
    # Generate post IDs (unique within a thread)
    post_ids = []
    thread_post_counters = {tid: 0 for tid in thread_ids}
    for tid in assigned_thread_ids:
        thread_post_counters[tid] += 1
        post_ids.append(f"{tid}-{thread_post_counters[tid]}")
    
    # Generate reply_to (some posts should reply to others in the same thread)
    reply_to = []
    for i, tid in enumerate(assigned_thread_ids):
        if thread_post_counters[tid] > 1 and random.random() < 0.7:  # 70% chance to reply
            # Reply to a random earlier post in the same thread
            earlier_posts = [j for j, x in enumerate(assigned_thread_ids[:i]) if x == tid]
            if earlier_posts:
                reply_to.append(post_ids[random.choice(earlier_posts)])
            else:
                reply_to.append(None)
        else:
            reply_to.append(None)
    
    # Generate other columns
    board_ids = [generate_board_id() for _ in range(num_rows)]
    subjects = [f"Subject {i}" if random.random() < 0.2 else None for i in range(num_rows)]
    authors = [f"Anonymous" for _ in range(num_rows)]
    country_codes = [generate_country_code() for _ in range(num_rows)]
    post_urls = [f"https://boards.4chan.org/{bid}/thread/{tid}#{pid}" 
                for bid, tid, pid in zip(board_ids, assigned_thread_ids, post_ids)]
    images = [random.randint(0, 2) for _ in range(num_rows)]  # 0-2 images per post
    post_texts = [generate_text() for _ in range(num_rows)]
    
    # Create DataFrame
    data = {
        'thread_id': assigned_thread_ids,
        'post_id': post_ids,
        'post_text': post_texts,
        'posted_date_time': timestamps,
        'reply_to': reply_to,
        'subject': subjects,
        'author': authors,
        'board_id': board_ids,
        'country_code': country_codes,
        'post_url': post_urls,
        'images': images
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} rows of synthetic data")
    logger.info(f"Date range: {df['posted_date_time'].min()} to {df['posted_date_time'].max()}")
    return df

async def add_synthetic_data_to_database(config, num_rows=1000, start_date=None, end_date=None):
    """
    Add synthetic data directly to the database.
    
    Args:
        config: ChanScope configuration
        num_rows: Number of rows to generate
        start_date: Start date for timestamps
        end_date: End date for timestamps
    """
    # Create data manager
    data_manager = ChanScopeDataManager(config)
    
    # Generate synthetic data
    df = generate_synthetic_data(num_rows, start_date, end_date)
    
    # Store in database
    logger.info("Storing synthetic data in database...")
    success = await data_manager.complete_data_storage.store_data(df)
    
    if success:
        logger.info("Synthetic data successfully stored in database")
        return True
    else:
        logger.error("Failed to store synthetic data in database")
        return False

async def run(args):
    """Run the script with parsed arguments."""
    # Parse dates
    if args.start_date:
        start_date = datetime.datetime.fromisoformat(args.start_date.replace('Z', '+00:00'))
    else:
        start_date = datetime.datetime.now() - datetime.timedelta(days=10)
    
    if args.end_date:
        end_date = datetime.datetime.fromisoformat(args.end_date.replace('Z', '+00:00'))
    else:
        end_date = datetime.datetime.now()
    
    # Load config from environment
    config = ChanScopeConfig.from_env()
    
    # Create data manager
    data_manager = ChanScopeDataManager(config)
    
    # Generate and add data
    success = await add_synthetic_data_to_database(
        config, 
        num_rows=args.num_rows,
        start_date=start_date,
        end_date=end_date
    )
    
    if success:
        # Always regenerate stratified sample to ensure it exists in the key-value store
        logger.info("Regenerating stratified sample...")
        stratify_success = await data_manager.ensure_stratified_sample(force_refresh=True)
        if stratify_success:
            logger.info("Stratified sample regenerated successfully")
            
            # Verify the stratified sample was stored in Replit's key-value store
            if os.environ.get('REPLIT_ENV') or os.environ.get('REPL_ID'):
                from config.replit import KeyValueStore
                kv_store = KeyValueStore()
                strat_sample = kv_store.get_stratified_sample()
                if strat_sample is not None and len(strat_sample) > 0:
                    logger.info(f"Verified stratified sample was stored in key-value store: {len(strat_sample)} rows")
                else:
                    logger.error("Failed to find stratified sample in key-value store")
        else:
            logger.error("Failed to regenerate stratified sample")
        
        # Check if embeddings should be regenerated
        if args.regenerate_embeddings:
            logger.info("Regenerating embeddings...")
            embedding_success = await data_manager.ensure_embeddings_generated(force_refresh=True)
            if embedding_success:
                logger.info("Embeddings regenerated successfully")
            else:
                logger.error("Failed to regenerate embeddings")
    
    # Return exit code
    return 0 if success else 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate and add synthetic data for testing")
    
    parser.add_argument(
        "--num-rows",
        type=int,
        default=1000,
        help="Number of rows to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for timestamps (ISO format, e.g., 2025-03-20T00:00:00). Defaults to 10 days ago."
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for timestamps (ISO format, e.g., 2025-03-30T23:59:59). Defaults to current time."
    )
    
    parser.add_argument(
        "--regenerate-stratified",
        action="store_true",
        help="Regenerate stratified sample after adding synthetic data"
    )
    
    parser.add_argument(
        "--regenerate-embeddings",
        action="store_true",
        help="Regenerate embeddings after adding synthetic data"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(run(args))
    sys.exit(exit_code) 