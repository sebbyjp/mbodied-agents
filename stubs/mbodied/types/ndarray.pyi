from _typeshed import Incomplete
from mypy.nodes import ARG_POS as ARG_POS, Argument as Argument, CallExpr as CallExpr, Var as Var
from mypy.plugin import AnalyzeTypeContext as AnalyzeTypeContext, Plugin
from mypy.types import Type as Type, TypeOfAny as TypeOfAny
from typing import TypeVar

NumpyArray: Incomplete
NumpyArrayFp32: Incomplete
T = TypeVar('T')

class NumpyArrayPlugin(Plugin):
    def get_type_analyze_hook(self, fullname: str): ...

def numpyarray_hook(ctx: AnalyzeTypeContext) -> Type: ...
def plugin(version: str): ...
