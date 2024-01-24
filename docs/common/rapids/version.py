# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import os.path
import re


def read_version():
    top = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))

    with open(os.path.join(top, "VERSION")) as f:
        if match := re.search(r"^(?P<major>[0-9][0-9])\.(?P<minor>[0-9][0-9])\.(?P<patch>[0-9][0-9])", f.read()):
            return match.group("major"), match.group("minor"), match.group("patch")
        raise RuntimeError("Could not parse VERSION file")


version_major, version_minor, version_patch = read_version()
version_major_minor = f"{version_major}.{version_minor}"
version = f"{version_major}.{version_minor}.{version_patch}"
