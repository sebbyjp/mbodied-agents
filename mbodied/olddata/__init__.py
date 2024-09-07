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

  # class MyModel(RootModel[T]):
                        #   root: T



# import sys
# import types

# import embdata
# import typing_extensions

# __getattr__ = embdata.__all__


# globals().update(
#     (name, getattr(embdata, name))
#     for name in [
#         "describe",
#         "episode",
#         "features",
#         "motion",
#         "ndarray",
#         "sample",
#         "sense",
#         "state",
#         "supervision",
#         "time",
#         "trajectory",
#         "units",
#         "utils",
#     ]
# )

# __all__ = [
#     "describe",
#     "episode",
# ]

# sys.modules[__name__] = sys.modules["embdata"]

# print(sys.modules)