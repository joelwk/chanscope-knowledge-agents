import re
import warnings
import string
from urllib.parse import urlparse
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from unicodedata import normalize
import pandas as pd
from pathlib import Path
from config.settings import Config

# Configure warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# Compile regex patterns
url_regex = re.compile(r'http\S+|www.\S+')
whitespace_regex = re.compile(r'\s+')
punctuation_regex = re.compile(f"([{string.punctuation}])")
non_alphanumeric_regex = re.compile(r'[^a-zA-Z0-9.,!?\' ]')

# Get the directory containing this file
current_dir = Path(__file__).parent
contraction_mapping_path = current_dir / 'contraction_mapping.json'

try:
    contraction_mapping = pd.read_json(contraction_mapping_path, typ='series').to_dict()
except FileNotFoundError:
    print(f"Warning: contraction_mapping.json not found at {contraction_mapping_path}")
    contraction_mapping = {}
    
wiki_markup_regex = re.compile(
    r'thumb\|[\dpx]*\|?|'
    r'right\|[\dpx]*\|?|'
    r'left\|[\dpx]*\|?|'
    r'center\|[\dpx]*\|?|'
    r'[\dpx]+\|'
)

def string_to_bool(string_value: str) -> bool:
    """Convert string to boolean."""
    return string_value.lower() in ['true', '1', 't', 'y', 'yes', 'on']

def pad_punctuation(s: str) -> str:
    """Add padding around punctuation if enabled."""
    processing_settings = Config.get_processing_settings()
    if processing_settings.get('padding_enabled', False):
        if not isinstance(s, str):
            return ""
        s = punctuation_regex.sub(r" \1 ", s)
        return whitespace_regex.sub(' ', s).strip()
    return s

def normalize_text(text: str) -> str:
    """Normalize text with configurable steps."""
    if isinstance(text, str):
        try:
            processing_settings = Config.get_processing_settings()
            
            # URL normalization
            text = url_regex.sub(lambda m: urlparse(m.group(0)).netloc.replace('www.', ''), text)
            text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            text = wiki_markup_regex.sub('', text)
            text = re.sub(r'\n\n.*?\n\n*?\n', ' ', text)
            text = text.replace('\n', ' ')
            text = ' '.join(BeautifulSoup(text, 'html.parser').stripped_strings)
            text = re.sub(r'>>\d+', ' ', text)
            
            # Remove wiki markup patterns
            text = re.sub(r'thumb\|\d*x\d*px\|right\|', '', text)
            text = re.sub(r'thumb\|\d*x\d*px\|', '', text)
            text = re.sub(r'thumb\|', '', text)
            text = re.sub(r'\d*x\d*px\|', '', text)
            text = re.sub(r'^\s*>+', '', text, flags=re.MULTILINE)
            
            # Apply contraction mapping if enabled
            if processing_settings.get('contraction_mapping_enabled', False):
                text = ' '.join(contraction_mapping.get(t, t) for t in text.split())
                
            # Remove non-alphanumeric if enabled
            if processing_settings.get('non_alpha_numeric_enabled', False):
                text = non_alphanumeric_regex.sub(' ', text)
                
            return whitespace_regex.sub(' ', text).strip()
        except ValueError:
            return text
    return text

def remove_whitespace(text: str) -> str:
    """Remove excess whitespace from text."""
    if isinstance(text, str):
        return " ".join(text.split())
    return text

def prepare_data(data: pd.DataFrame, input_col: str, clean_col: str) -> pd.DataFrame:
    """Prepare data by applying text normalization."""
    data[input_col] = data[input_col].astype(str)
    data[clean_col] = data[input_col].apply(normalize_text).apply(remove_whitespace).apply(pad_punctuation)
    return data[data[clean_col].notnull() & data[clean_col].str.strip().astype(bool)]