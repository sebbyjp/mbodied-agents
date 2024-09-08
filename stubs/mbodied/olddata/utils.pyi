from _typeshed import Incomplete
from datasets import Features

def to_features(indict, image_keys: Incomplete | None = None, exclude_keys: Incomplete | None = None, prefix: str = '') -> Features:
    """Convert a dictionary to a Datasets Features object.

    Args:
        indict (dict): The dictionary to convert.
        image_keys (dict): A dictionary of keys that should be treated as images.
        exclude_keys (set): A set of full-path-keys to exclude.
        prefix (str): A prefix to add to the keys.
    """
def infer_features(example) -> Features:
    """Infer Hugging Face Datasets Features from an example."""
