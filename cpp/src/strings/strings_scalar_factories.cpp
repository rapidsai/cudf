/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
// Create a strings-type column from array of pointer/size pairs
std::unique_ptr<scalar> make_string_scalar(std::string const& string,
                                           rmm::cuda_stream_view stream,
                                           cudf::memory_resources resources)
{
  auto s = new string_scalar(string, true, stream, resources);
  return std::unique_ptr<scalar>(s);
}

}  // namespace cudf
