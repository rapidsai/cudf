/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.cuh"

namespace cudf::binops::compiled {
template void apply_binary_op<ops::NullMin>(mutable_column_view&,
                                            column_view const&,
                                            column_view const&,
                                            bool is_lhs_scalar,
                                            bool is_rhs_scalar,
                                            rmm::cuda_stream_view);
}  // namespace cudf::binops::compiled
