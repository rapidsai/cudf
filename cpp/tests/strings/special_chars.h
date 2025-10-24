/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf::test {

// special new-line characters for use with regex_flags::EXT_NEWLINE
#define NEXT_LINE           "\xC2\x85"
#define LINE_SEPARATOR      "\xE2\x80\xA8"
#define PARAGRAPH_SEPARATOR "\xE2\x80\xA9"

}  // namespace cudf::test
