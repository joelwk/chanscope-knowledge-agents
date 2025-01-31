import pandas as pd
import re
import logging
import asyncio
from pathlib import Path
import gc

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReferenceProcessor:
    """Processes data to extract and analyze thread references."""
    
    def __init__(self, output_dir: str = 'data'):
        """Initialize the processor with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_pattern = re.compile(r'&gt;&gt;(\d+)')
        logger.info(f"Initialized ReferenceProcessor with output_dir: {self.output_dir.absolute()}")
        
    def extract_references(self, comment: str) -> list:
        """Extract thread references from a comment."""
        if not isinstance(comment, str):
            return []
        return self.reference_pattern.findall(comment)
    
    def create_reference_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a DataFrame with reference information."""
        rows = []
        for idx, row in df.iterrows():
            comment = row.get('posted_comment', '')  # Using text_clean as per the data schema
            thread_id = row.get('thread_id')
            posted_date_time = row.get('posted_date_time')
            
            if not all([comment, thread_id, posted_date_time]):
                continue
                
            references = self.extract_references(comment)
            
            for ref_id in references:
                referenced_row = df[df['thread_id'] == int(ref_id)]
                if not referenced_row.empty:
                    referenced_comment = referenced_row.iloc[0]['text_clean']
                    referenced_posted_date_time = referenced_row.iloc[0]['posted_date_time']
                else:
                    referenced_comment = None
                    referenced_posted_date_time = None
                
                rows.append({
                    'posted_date_time': posted_date_time,
                    'text_clean': comment,
                    'comment_id': ref_id,
                    'reference_text': referenced_comment,
                    'reference_id': thread_id,
                    'reference_posted_date_time': referenced_posted_date_time
                })
        
        return pd.DataFrame(rows)
    
    def prepare_data(self, df: pd.DataFrame, source_col: str, target_col: str) -> pd.DataFrame:
        """Prepare data by processing text columns."""
        logger.info(f"Preparing data: source_col={source_col}, target_col={target_col}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        df = df.copy()
        if source_col not in df.columns:
            raise KeyError(f"Source column '{source_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            
        df[target_col] = df[source_col].fillna("None")
        return df
    
    async def process_and_save_data(self, df: pd.DataFrame):
        """Process existing data and save reference information."""
        try:
            logger.info(f"Starting data processing with {len(df)} rows")
            logger.info(f"Initial columns: {df.columns.tolist()}")
            
            # Map column names if needed
            column_mapping = {
                'posted_comment': 'text_clean',  # Map posted_comment to text_clean if needed
            }
            
            # Rename columns if they exist
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    logger.info(f"Renaming column {old_col} to {new_col}")
                    df = df.rename(columns={old_col: new_col})
            
            logger.info(f"Columns after mapping: {df.columns.tolist()}")
            
            # Process references
            reference_df = self.create_reference_dataframe(df)
            logger.info(f"Created reference DataFrame with {len(reference_df)} rows")
            
            # Apply data preparation steps
            processed_df = self.prepare_data(reference_df, 'text_clean', 'response_comment')
            processed_df = self.prepare_data(processed_df, 'reference_text', 'seed_comment')
            
            # Filter and organize final dataset
            final_df = processed_df[processed_df['seed_comment'] != "None"].drop(columns=['text_clean', 'reference_text'])
            final_df = final_df[[
                'reference_posted_date_time',
                'seed_comment',
                'comment_id',
                'posted_date_time',
                'response_comment',
                'reference_id'
            ]]
            
            # Save processed data
            output_path = self.output_dir / 'processed_references.csv'
            final_df.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to {output_path}")
            
            # Clean up
            del reference_df, processed_df, final_df
            gc.collect()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            raise

async def process_references(output_dir: str = 'data') -> Path:
    """Main entry point for reference processing."""
    try:
        # Initialize processor
        processor = ReferenceProcessor(output_dir)
        logger.info(f"Initialized ReferenceProcessor with output_dir: {output_dir}")
        
        # Load existing stratified_sample.csv
        stratified_sample_path = Path(output_dir) / 'stratified_sample.csv'
        logger.info(f"Looking for stratified_sample.csv at: {stratified_sample_path.absolute()}")
        
        # Check if file exists and is readable
        if not stratified_sample_path.exists():
            logger.error(f"File not found at {stratified_sample_path.absolute()}")
            raise FileNotFoundError(f"Could not find stratified_sample.csv in {output_dir}")
        
        # Check file size
        file_size = stratified_sample_path.stat().st_size
        logger.info(f"Found file of size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("File exists but is empty")
            raise ValueError("stratified_sample.csv exists but is empty")
            
        # Try to read the first few lines of the file to check content and encoding
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            file_content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(stratified_sample_path, 'r', encoding=encoding) as f:
                        file_content = [next(f) for _ in range(5)]  # Get first 5 lines
                        used_encoding = encoding
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error reading with {encoding}: {e}")
                    continue
            
            if file_content is None:
                raise ValueError("Could not read file with any supported encoding")
                
            logger.info(f"Successfully read file with encoding: {used_encoding}")
            logger.info(f"File header: {file_content[0].strip()}")
            logger.info(f"Sample lines:\n" + "\n".join(line.strip() for line in file_content[1:]))
            
            # Read the full dataset with the working encoding
            logger.info(f"Reading CSV with encoding {used_encoding}...")
            df = pd.read_csv(
                stratified_sample_path,
                encoding=used_encoding,
                on_bad_lines='warn',
                low_memory=False
            )
            
            if df.empty:
                logger.error("DataFrame is empty after reading")
                raise ValueError("No data was read from the CSV file")
            
            logger.info("Successfully read CSV file")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"Column dtypes:\n{df.dtypes}")
            logger.info(f"First few rows:\n{df.head().to_string()}")
            
            df['posted_date_time'] = pd.to_datetime(df['posted_date_time'], utc=True)
            logger.info(f"Loaded and processed {len(df)} rows from stratified_sample.csv")
            
            # Process the data
            return await processor.process_and_save_data(df)
            
        except Exception as e:
            logger.error(f"Error reading file: {str(e)}")
            logger.error(f"Full error details: {str(e.__class__.__name__)}: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Process references failed: {str(e)}")
        raise

if __name__ == "__main__":
    # When running the script directly, execute the async function
    output_path = asyncio.run(process_references())
    print(f"Processed data saved to: {output_path}") 