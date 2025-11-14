/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/stream_compaction/stream_compaction_common.hpp>

#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

cudf::duplicate_keep_option get_keep(std::string const& keep_str)
{
  if (keep_str == "any") {
    return cudf::duplicate_keep_option::KEEP_ANY;
  } else if (keep_str == "first") {
    return cudf::duplicate_keep_option::KEEP_FIRST;
  } else if (keep_str == "last") {
    return cudf::duplicate_keep_option::KEEP_LAST;
  } else if (keep_str == "none") {
    return cudf::duplicate_keep_option::KEEP_NONE;
  } else {
    CUDF_FAIL("Unsupported keep option.");
  }
}
