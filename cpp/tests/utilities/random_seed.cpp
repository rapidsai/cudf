/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/utilities/export.hpp>

#include <cstdint>

namespace cudf {
namespace test {
namespace detail {

/**
 * @copydoc cudf::test::detail::random_generator_incrementing_seed()
 */
CUDF_EXPORT uint64_t random_generator_incrementing_seed()
{
  static uint64_t seed = 0;
  return ++seed;
}

}  // namespace detail
}  // namespace test
}  // namespace cudf
