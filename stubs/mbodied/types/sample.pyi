import numpy as np
import torch
from _typeshed import Incomplete
from gymnasium import spaces
from mbodied.data import features as features
from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from typing import Any, Literal
from typing_extensions import Annotated

Flattenable: Incomplete

class Sample(BaseModel):
    '''A base model class for serializing, recording, and manipulating arbitray data.

    It was designed to be extensible, flexible, yet strongly typed. In addition to
    supporting any json API out of the box, it can be used to represent
    arbitrary action and observation spaces in robotics and integrates seemlessly with H5, Gym, Arrow,
    PyTorch, DSPY, numpy, and HuggingFace.

    Methods:
        schema: Get a simplified json schema of your data.
        to: Convert the Sample instance to a different container type:
            -
        default_value: Get the default value for the Sample instance.
        unflatten: Unflatten a one-dimensional array or dictionary into a Sample instance.
        flatten: Flatten the Sample instance into a one-dimensional array or dictionary.
        space_for: Default Gym space generation for a given value.
        init_from: Initialize a Sample instance from a given value.
        from_space: Generate a Sample instance from a Gym space.
        pack_from: Pack a list of samples into a single sample with lists for attributes.
        unpack: Unpack the packed Sample object into a list of Sample objects or dictionaries.
        dict: Return the Sample object as a dictionary with None values excluded.
        model_field_info: Get the FieldInfo for a given attribute key.
        space: Return the corresponding Gym space for the Sample instance based on its instance attributes.
        random_sample: Generate a random Sample instance based on its instance attributes.

    Examples:
        >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
        >>> flat_list = sample.flatten()
        >>> print(flat_list)
        [1, 2, 3, 4, 5]
        >>> schema = sample.schema()
        {\'type\': \'object\', \'properties\': {\'x\': {\'type\': \'number\'}, \'y\': {\'type\': \'number\'}, \'z\': {\'type\': \'object\', \'properties\': {\'a\': {\'type\': \'number\'}, \'b\': {\'type\': \'number\'}}}, \'extra_field\': {\'type\': \'number\'}}}
        >>> unflattened_sample = Sample.unflatten(flat_list, schema)
        >>> print(unflattened_sample)
        Sample(x=1, y=2, z={\'a\': 3, \'b\': 4}, extra_field=5)
    '''
    __doc__: str
    model_config: ConfigDict
    def __init__(self, datum: Incomplete | None = None, **data) -> None:
        """Accepts an arbitrary datum as well as keyword arguments."""
    def __hash__(self) -> int:
        """Return a hash of the Sample instance."""
    def dict(self, exclude_none: bool = True, exclude: set[str] = None) -> dict[str, Any]:
        """Return the Sample object as a dictionary with None values excluded.

        Args:
            exclude_none (bool, optional): Whether to exclude None values. Defaults to True.
            exclude (set[str], optional): Set of attribute names to exclude. Defaults to None.

        Returns:
            Dict[str, Any]: Dictionary representation of the Sample object.
        """
    @classmethod
    def unflatten(cls, one_d_array_or_dict, schema: Incomplete | None = None) -> Sample:
        '''Unflatten a one-dimensional array or dictionary into a Sample instance.

        If a dictionary is provided, its keys are ignored.

        Args:
            one_d_array_or_dict: A one-dimensional array or dictionary to unflatten.
            schema: A dictionary representing the JSON schema. Defaults to using the class\'s schema.

        Returns:
            Sample: The unflattened Sample instance.

        Examples:
            >>> sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
            >>> flat_list = sample.flatten()
            >>> print(flat_list)
            [1, 2, 3, 4, 5]
            >>> Sample.unflatten(flat_list, sample.schema())
            Sample(x=1, y=2, z={\'a\': 3, \'b\': 4}, extra_field=5)
        '''
    def flatten(self, output_type: Flattenable = 'dict', non_numerical: Literal['ignore', 'forbid', 'allow'] = 'allow') -> dict[str, Any] | np.ndarray | torch.Tensor | list: ...
    @staticmethod
    def obj_to_schema(value: Any) -> dict:
        """Generates a simplified JSON schema from a dictionary.

        Args:
            value (Any): An object to generate a schema for.

        Returns:
            dict: A simplified JSON schema representing the structure of the dictionary.
        """
    def schema(self, resolve_refs: bool = True, include_descriptions: bool = False) -> dict:
        """Returns a simplified json schema.

        Removing additionalProperties,
        selecting the first type in anyOf, and converting numpy schema to the desired type.
        Optionally resolves references.

        Args:
            schema (dict): A dictionary representing the JSON schema.
            resolve_refs (bool): Whether to resolve references in the schema. Defaults to True.
            include_descriptions (bool): Whether to include descriptions in the schema. Defaults to False.

        Returns:
            dict: A simplified JSON schema.
        """
    @classmethod
    def read(cls, data: Any) -> Sample:
        """Read a Sample instance from a JSON string or dictionary or path.

        Args:
            data (Any): The JSON string or dictionary to read.

        Returns:
            Sample: The read Sample instance.
        """
    def to(self, container: Any) -> Any:
        """Convert the Sample instance to a different container type.

        Args:
            container (Any): The container type to convert to. Supported types are
            'dict', 'list', 'np', 'pt' (pytorch), 'space' (gym.space),
            'schema', 'json', 'hf' (datasets.Dataset) and any subtype of Sample.

        Returns:
            Any: The converted container.
        """
    @classmethod
    def default_value(cls) -> Sample:
        """Get the default value for the Sample instance.

        Returns:
            Sample: The default value for the Sample instance.
        """
    @classmethod
    def space_for(cls, value: Any, max_text_length: int = 1000, info: Annotated = None) -> spaces.Space:
        """Default Gym space generation for a given value.

        Only used for subclasses that do not override the space method.
        """
    @classmethod
    def init_from(cls, d: Any, pack: bool = False) -> Sample: ...
    @classmethod
    def from_flat_dict(cls, flat_dict: dict[str, Any], schema: dict = None) -> Sample:
        """Initialize a Sample instance from a flattened dictionary."""
    @classmethod
    def from_space(cls, space: spaces.Space) -> Sample:
        """Generate a Sample instance from a Gym space."""
    @classmethod
    def pack_from(cls, samples: list[Sample | dict]) -> Sample:
        """Pack a list of samples into a single sample with lists for attributes.

        Args:
            samples (List[Union[Sample, Dict]]): List of samples or dictionaries.

        Returns:
            Sample: Packed sample with lists for attributes.
        """
    def unpack(self, to_dicts: bool = False) -> list[Sample | dict]:
        """Unpack the packed Sample object into a list of Sample objects or dictionaries."""
    @classmethod
    def default_space(cls) -> spaces.Dict:
        """Return the Gym space for the Sample class based on its class attributes."""
    @classmethod
    def default_sample(cls, output_type: str = 'Sample') -> Sample | dict[str, Any]:
        '''Generate a default Sample instance from its class attributes. Useful for padding.

        This is the "no-op" instance and should be overriden as needed.
        '''
    def model_field_info(self, key: str) -> FieldInfo:
        """Get the FieldInfo for a given attribute key."""
    def space(self) -> spaces.Dict:
        """Return the corresponding Gym space for the Sample instance based on its instance attributes. Omits None values.

        Override this method in subclasses to customize the space generation.
        """
    def random_sample(self) -> Sample:
        """Generate a random Sample instance based on its instance attributes. Omits None values.

        Override this method in subclasses to customize the sample generation.
        """
