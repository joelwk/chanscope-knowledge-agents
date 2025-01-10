import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def split_dataframe(df, fraction=None, sample_size=None, stratify_column=None, 
                   save_directory=None, seed=None, file_format='csv'):
    """
    Split a DataFrame into stratified subsets based on a fraction or sample size,
    with an optional feature to save the subsets to a directory.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - fraction (float, optional): Fraction of the data to include in each split. 
                               Defaults to None.
    - sample_size (int, optional): Number of rows to include in each split. 
                                Defaults to None.
    - stratify_column (str, optional): Column name to stratify the splits by. 
                                    Must be present in the DataFrame. Defaults to None.
    - save_directory (str, optional): Directory path to save the subsets. 
                                   Defaults to None.
    - seed (int, optional): Random seed for reproducibility. Defaults to None.
    - file_format (str, optional): File format for saving (e.g., 'csv', 'pickle', 'excel'). 
                                Defaults to 'csv'.
    
    Returns:
    - list of pd.DataFrames: A list of stratified subsets of the original DataFrame.
    """
    # Input validation
    if fraction is not None and sample_size is not None:
        raise ValueError("Cannot provide both 'fraction' and 'sample_size'. Choose one.")
        
    if fraction is None and sample_size is None:
        raise ValueError("Must provide either 'fraction' or 'sample_size'.")

    if stratify_column is not None and stratify_column not in df.columns:
        raise ValueError(f"'{stratify_column}' is not a column in the DataFrame.")

    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Reset index to avoid issues with index alignment
    df = df.reset_index(drop=True)

    # Calculate the number of splits based on the fraction or sample size
    if fraction is not None:
        sample_size = int(len(df) * fraction)
    
    num_splits = int(np.floor(len(df) / sample_size))
    remaining_rows = len(df) % sample_size

    # Initialize an empty list to store the subsets
    subsets = []

    # Stratified Split
    if stratify_column:
        remaining_df = df.copy()
        while len(remaining_df) >= sample_size:
            # Get stratified sample
            stratified_sample = remaining_df.groupby(stratify_column, group_keys=False) \
                .apply(lambda x: x.sample(min(len(x), int(np.ceil(sample_size * len(x) / len(remaining_df)))), 
                                        random_state=seed))
            
            # Ensure we don't exceed sample_size
            if len(stratified_sample) > sample_size:
                stratified_sample = stratified_sample.sample(sample_size, random_state=seed)
            
            subsets.append(stratified_sample)
            
            # Remove sampled indices from remaining_df
            remaining_df = remaining_df.drop(stratified_sample.index)
        
        # Handle remaining rows if any
        if len(remaining_df) > 0:
            subsets.append(remaining_df)
    else:
        # Random split
        df_shuffled = df.sample(frac=1, random_state=seed)
        for i in range(num_splits):
            subsets.append(df_shuffled.iloc[i*sample_size:(i+1)*sample_size].copy())
        
        # Handle remaining rows
        if remaining_rows > 0:
            subsets.append(df_shuffled.tail(remaining_rows).copy())

    # Save subsets if directory provided
    if save_directory:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, subset in enumerate(subsets):
            filename = f"dataset_{i+1}_subset"
            filepath = save_dir / f"{filename}.{file_format}"
            
            if file_format == 'csv':
                subset.to_csv(filepath, index=False)
            elif file_format == 'pickle':
                subset.to_pickle(filepath)
            elif file_format == 'excel':
                subset.to_excel(filepath, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")

    return subsets