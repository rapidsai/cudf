/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

constexpr uint32_t NEW_CP_MASK = 0x1f'ffffu;

constexpr uint32_t MULTICHAR_SHIFT = 23;
constexpr uint32_t MULTICHAR_MASK  = 1;

constexpr uint32_t TOKEN_CAT_SHIFT                = 24;
constexpr uint32_t TOKEN_CAT_MASK                 = 7;
constexpr uint32_t TOKEN_CAT_ADD_SPACE            = 0;
constexpr uint32_t TOKEN_CAT_ADD_SPACE_IF_LOWER   = 1;
constexpr uint32_t TOKEN_CAT_REMOVE_CHAR          = 2;
constexpr uint32_t TOKEN_CAT_REMOVE_CHAR_IF_LOWER = 3;
constexpr uint32_t TOKEN_CAT_ALWAYS_REPLACE       = 4;

constexpr uint32_t SPACE_CODE_POINT = 32;
constexpr uint32_t MAX_NEW_CHARS    = 3;

using codepoint_metadata_type = uint32_t;
using aux_codepoint_data_type = uint64_t;
