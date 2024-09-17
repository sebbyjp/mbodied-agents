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
from pathlib import Path
from time import sleep, time
from typing import Generic, TypeVar

from mbodied.robots import Robot
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

ObsT, StaT, ActT, SupervisionT = (
    TypeVar("ObsT", bound=Sample),
    TypeVar("StateT", bound=Sample),
    TypeVar("ActionT", bound=Sample),
    TypeVar("SupervisionT"),
)
