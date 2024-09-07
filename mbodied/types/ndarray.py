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

import pydantic_numpy.typing as pnd

"""Pydantic Numpy Array for serialization/deserialization."""

NumpyArray = pnd.NpNDArray
NumpyArrayFp32 = pnd.NpNDArrayFp32

from typing import Generic, TypeVar, Tuple

# Define a type variable for the data type of the array
T = TypeVar('T')

from mypy.plugin import Plugin, AnalyzeTypeContext
from mypy.types import Type, TypeOfAny
from mypy.nodes import ARG_POS, Argument, CallExpr, TypeInfo, Var

# The hook that processes NumpyArray type annotations
class NumpyArrayPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str):
        if fullname == '__main__.NumpyArray':  # Replace with the actual path to NumpyArray
            return numpyarray_hook
        return None

# Hook that processes the type annotation
def numpyarray_hook(ctx: AnalyzeTypeContext) -> Type:
    # Parse the shape and dtype from the annotation
    shape_args = ctx.type.args[0]  # First argument is shape
    dtype_arg = ctx.type.args[1]    # Second argument is dtype
    
    # Check if shape and dtype are valid (you can enforce more complex rules here)
    if isinstance(shape_args, tuple) and isinstance(dtype_arg, TypeInfo):
        # You can perform type validation here if needed
        pass
    
    # Return the processed type
    return ctx.api.named_type('__main__.NumpyArray', [shape_args, dtype_arg])

# Register the plugin
def plugin(version: str):
    return NumpyArrayPlugin