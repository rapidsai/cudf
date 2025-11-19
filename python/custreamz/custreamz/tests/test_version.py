# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import custreamz


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(custreamz.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(custreamz.__version__, str)
    assert len(custreamz.__version__) > 0
