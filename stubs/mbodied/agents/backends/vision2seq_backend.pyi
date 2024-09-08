import abc
from typing import Literal

import torch
from _typeshed import Incomplete

from mbodied.agents.backends.backend import Backend as Backend
from mbodied.agents.backends.serializer import Serializer as Serializer
from mbodied.types.sense.vision import Image as Image

class Vision2SeqBackendSerializer(Serializer): ...

class Vision2SeqBackend(Backend, metaclass=abc.ABCMeta):
    """Vision2SeqBackend backend that runs locally to generate robot actions.

    Beware of the memory requirements of 7B+ parameter models like OpenVLA.

    Attributes:
        model_id (str): The model to use for the OpenVLA backend.
        device (torch.device): The device to run the model on.
        torch_dtype (torch.dtype): The torch data type to use.
        processor (AutoProcessor): The processor for the model.
        model (AutoModelForVision2Seq): The model for the OpenVLA backend.
    """
    DEFAULT_DEVICE: Incomplete
    ATTN_IMPLEMENTATION: Incomplete
    DTYPE: Incomplete
    model_id: Incomplete
    device: Incomplete
    torch_dtype: Incomplete
    processor: Incomplete
    model: Incomplete
    def __init__(self, model_id: str = 'openvla/openvla-7b', attn_implementation: Literal['flash_attention_2', 'eager'] = ..., torch_dtype: torch.dtype = ..., device: torch.device = ..., **kwargs) -> None: ...
    def predict(self, instruction: str, image: Image, unnorm_key: str = 'bridge_orig') -> str: ...
