import numpy as np
import os
import logging
from pathlib import Path
from typing import Iterator, Optional
import pandas as pd

logger = logging.getLogger(__name__)

class StreamingStratifier:
    """A streaming implementation of stratified sampling."""

    def __init__(self, fraction: Optional[float] = None, 
                 sample_size: Optional[int] = None,
                 stratify_column: Optional[str] = None,
                 seed: Optional[int] = None):
        """Initialize the streaming stratifier.

        Args:
            fraction: Fraction of data to sample in each split
            sample_size: Number of rows to include in each split
            stratify_column: Column to stratify by
            seed: Random seed for reproducibility
        """
        if fraction is not None and sample_size is not None:
            raise ValueError("Cannot provide both 'fraction' and 'sample_size'")
        if fraction is None and sample_size is None:
            raise ValueError("Must provide either 'fraction' or 'sample_size'")

        self.fraction = fraction
        self.sample_size = sample_size
        self.stratify_column = stratify_column
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Initialize state
        self.current_split = []
        self.current_split_size = 0
        self.total_processed = 0
        self.strata_counts = {}

    def _calculate_target_size(self, chunk_size: int) -> int:
        """Calculate target size for current chunk based on fraction or sample_size."""
        if self.fraction is not None:
            return int(chunk_size * self.fraction)
        return min(self.sample_size, chunk_size)

    def process_chunk(self, chunk: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Process a chunk of data and yield complete splits when ready.

        Args:
            chunk: DataFrame chunk to process

        Yields:
            Complete splits of the data when they reach the target size
        """
        if self.stratify_column and self.stratify_column not in chunk.columns:
            raise ValueError(f"Stratify column '{self.stratify_column}' not in chunk")

        target_size = self._calculate_target_size(len(chunk))

        if self.stratify_column:
            # Update strata counts
            current_strata = chunk[self.stratify_column].value_counts()
            for stratum, count in current_strata.items():
                self.strata_counts[stratum] = self.strata_counts.get(stratum, 0) + count

            # Calculate sampling weights for each stratum
            total_rows = sum(self.strata_counts.values())
            strata_weights = {
                stratum: count / total_rows 
                for stratum, count in self.strata_counts.items()
            }

            # Sample from each stratum proportionally
            sampled_indices = []
            for stratum in chunk[self.stratify_column].unique():
                stratum_mask = chunk[self.stratify_column] == stratum
                stratum_size = int(target_size * strata_weights.get(stratum, 0))
                if stratum_size > 0:
                    stratum_indices = chunk[stratum_mask].index
                    if len(stratum_indices) > stratum_size:
                        sampled_indices.extend(
                            self.rng.choice(stratum_indices, size=stratum_size, replace=False)
                        )
                    else:
                        sampled_indices.extend(stratum_indices)

            current_sample = chunk.loc[sampled_indices]
        else:
            # Simple random sampling
            if len(chunk) > target_size:
                current_sample = chunk.sample(n=target_size, random_state=self.rng)
            else:
                current_sample = chunk

        self.current_split.append(current_sample)
        self.current_split_size += len(current_sample)
        self.total_processed += len(chunk)

        # Yield complete splits
        if self.sample_size and self.current_split_size >= self.sample_size:
            combined_split = pd.concat(self.current_split, ignore_index=True)
            if len(combined_split) > self.sample_size:
                combined_split = combined_split.head(self.sample_size)
            yield combined_split
            self.current_split = []
            self.current_split_size = 0

    def get_remaining_split(self) -> Optional[pd.DataFrame]:
        """Get any remaining data as a final split."""
        if self.current_split:
            return pd.concat(self.current_split, ignore_index=True)
        return None

def stream_stratified_split(data_iterator: Iterator[pd.DataFrame], 
                          fraction: Optional[float] = None,
                          sample_size: Optional[int] = None,
                          stratify_column: Optional[str] = None,
                          seed: Optional[int] = None) -> Iterator[pd.DataFrame]:
    """Stream and stratify data in chunks.

    Args:
        data_iterator: Iterator yielding DataFrame chunks
        fraction: Fraction of data to include in each split
        sample_size: Number of rows to include in each split
        stratify_column: Column to stratify by
        seed: Random seed for reproducibility

    Yields:
        DataFrame splits of the specified size
    """
    stratifier = StreamingStratifier(
        fraction=fraction,
        sample_size=sample_size,
        stratify_column=stratify_column,
        seed=seed
    )

    for chunk in data_iterator:
        yield from stratifier.process_chunk(chunk)

    # Yield any remaining data
    remaining = stratifier.get_remaining_split()
    if remaining is not None:
        yield remaining

# Keep the original function for backwards compatibility
def split_dataframe(df, fraction=None, sample_size=None, stratify_column=None, 
                   save_directory=None, seed=None, file_format='csv'):
    """Original function maintained for backwards compatibility."""
    chunks = [df]  # Convert full DataFrame to single-chunk iterator
    splits = list(stream_stratified_split(
        chunks, fraction, sample_size, stratify_column, seed
    ))

    if save_directory:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, split in enumerate(splits):
            filename = f"dataset_{i+1}_subset"
            filepath = save_dir / f"{filename}.{file_format}"

            if file_format == 'csv':
                split.to_csv(filepath, index=False)
            elif file_format == 'pickle':
                split.to_pickle(filepath)
            elif file_format == 'excel':
                split.to_excel(filepath, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    return splits