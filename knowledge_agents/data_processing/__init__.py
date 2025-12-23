from .sampler import Sampler

def load_all_csv_data_from_s3(*args, **kwargs):
    """Lazily import S3 helpers to avoid heavy deps at import time."""
    from .cloud_handler import load_all_csv_data_from_s3 as _load_all_csv_data_from_s3
    return _load_all_csv_data_from_s3(*args, **kwargs)

__all__ = [
    'Sampler',
    'load_all_csv_data_from_s3'
]
