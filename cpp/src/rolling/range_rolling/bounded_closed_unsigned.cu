/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dispatch.cuh"

namespace cudf::detail {

std::unique_ptr<column> dispatch_range_window(bounded_closed window,
                                              rolling::unsigned_integral_orderby,
                                              rolling::range_window_dispatch_args const& args)
{
  return rolling::dispatch_range_window_by_type<rolling::dispatch_unsigned_integral_orderby>(window,
                                                                                             args);
}

}  // namespace cudf::detail
