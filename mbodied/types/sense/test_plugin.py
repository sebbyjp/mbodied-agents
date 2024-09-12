
# test_plugin.py
from typing import Any
from mbodied.types.ndarray import NumpyArray
# Assuming NumpyArray is imported or defined somewhere
mask: 'NumpyArray[2, 2, float]'

def test_mask(mask: 'NumpyArray[2, 2, float]') -> Any:
    return mask

  
test_mask()