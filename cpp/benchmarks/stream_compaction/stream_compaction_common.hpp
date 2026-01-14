/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/stream_compaction.hpp>

cudf::duplicate_keep_option get_keep(std::string const& keep_str);
