# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.resources

__version__ = (
    importlib.resources.files(__package__)
    .joinpath("VERSION")
    .read_text()
    .strip()
)
try:
    __git_commit__ = (
        importlib.resources.files(__package__)
        .joinpath("GIT_COMMIT")
        .read_text()
        .strip()
    )
except FileNotFoundError:
    __git_commit__ = ""

__all__ = ["__git_commit__", "__version__"]
