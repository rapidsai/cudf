# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.libcudf.strings.regex_flags import \
    regex_flags as RegexFlags  # no-cython-lint

__all__ = ["RegexFlags"]

RegexFlags.__str__ = RegexFlags.__repr__
