import datasets.features.features
from _typeshed import Incomplete

def to_features(indict, image_keys: Incomplete | None = ..., exclude_keys: Incomplete | None = ..., prefix: str = ...) -> datasets.features.features.Features:
    """Convert a dictionary to a Datasets Features object.

        Args:
            indict (dict): The dictionary to convert.
            image_keys (dict): A dictionary of keys that should be treated as images.
            exclude_keys (set): A set of full-path-keys to exclude.
            prefix (str): A prefix to add to the keys.
    """
def infer_features(example) -> datasets.features.features.Features:
    """Infer Hugging Face Datasets Features from an example."""
