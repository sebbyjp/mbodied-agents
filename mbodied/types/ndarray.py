# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import pydantic_numpy.typing as pnd

# """Pydantic Numpy Array for serialization/deserialization."""

# from typing import Generic, TypeVar, Tuple

# # Define a type variable for the data type of the array
# T = TypeVar('T')

# from mypy.nodes import ARG_POS, Argument, CallExpr, TypeInfo, Var  # noqa: E402
# from mypy.plugin import AnalyzeTypeContext, Plugin
# from mypy.types import Type, TypeOfAny


# # The hook that processes NumpyArray type annotations
# class NumpyArrayPlugin(Plugin):
#     def get_type_analyze_hook(self, fullname: str):
#         if fullname == '__main__.NumpyArray':  # Replace with the actual path to NumpyArray
#             return numpyarray_hook
#         return None

# # Hook that processes the type annotation
# def numpyarray_hook(ctx: AnalyzeTypeContext) -> Type:
#     # Parse the shape and dtype from the annotation
#     shape_args = ctx.type.args[:-1]
#     dtype_arg = ctx.type.args[-1]

#     if isinstance(dtype_arg, TypeOfAny):
#         # If the dtype is Any, we can't determine the type of the array
#         ctx.api.fail('NumpyArray type must have a valid dtype', ctx.context)
#         return ctx.api.named_type('builtins.Any')
#     # Check if shape and dtype are valid (you can enforce more complex rules here)
#     if isinstance(shape_args, tuple) and isinstance(dtype_arg, TypeInfo):
#         # You can perform type validation here if needed
#         pass
    
#     # Return the processed type
#     return ctx.api.named_type('__main__.NumpyArray', [shape_args, dtype_arg])

# Register the plugin
# def plugin(version: str):
#     return NumpyArrayPlugin

from typing import TypeVarTuple, Generic

from mypy.plugin import Plugin, AnalyzeTypeContext
from mypy.types import UnboundType, Type
from typing import Callable
from mypy.plugin import Plugin, AnalyzeTypeContext
from mypy.types import Type, UnboundType, Instance
from typing import Callable

Ts = TypeVarTuple('T')
class NumpyArray(Generic[*Ts]):
    def __init__(self, shape: tuple[int, ...], dtype: type):
        self.shape = shape
        self.dtype = dtype
    
    def __class_getitem__(cls, items):
        cls.__repr__ = lambda self: f"{cls.__name__}{items}"
        cls.__str__ = lambda self: f"{cls.__name__}{items}"
        return cls

class NumpyArrayPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str) -> Callable[[AnalyzeTypeContext], Type] | None:
        if fullname == "NumpyArray":
            return self.analyze_numpy_array_type
        return None

    def analyze_numpy_array_type(self, ctx: AnalyzeTypeContext) -> Type:
        unbound_type = ctx.type  # This is the unprocessed NumpyArray[...] type

        # Check if NumpyArray has valid arguments like NumpyArray[224, 224, 3, float]
        if isinstance(unbound_type, UnboundType):
            args = unbound_type.args
            
            # Check if all arguments are valid, integers or types like float
            for arg in args[:-1]:  # Skip the last argument (which is assumed to be dtype)
                if isinstance(arg, UnboundType) and not arg.name.isdigit():
                    # If we encounter non-integer dimensions (excluding dtype), raise an error
                    ctx.api.fail(f"NumpyArray dimensions must be integers", ctx.context)
                    return ctx.api.named_type('builtins.object')

            # Return the unmodified type to preserve the clean NumpyArray[...] format
            return unbound_type

        # If anything is wrong, report an error
        ctx.api.fail(f"NumpyArray expects valid dimensions and dtype", ctx.context)
        return ctx.api.named_type('builtins.object')



def plugin(version: str):
    return NumpyArrayPlugin