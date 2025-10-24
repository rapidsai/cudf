/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/pair.h>

#include <string>
#include <vector>

class StringsFactoryTest : public cudf::test::BaseFixture {};

using string_pair = thrust::pair<char const*, cudf::size_type>;

TEST_F(StringsFactoryTest, StringConstructionFromPairs)
{
  auto const stream = cudf::test::get_default_stream();

  auto const h_data = std::vector<char>{'a', 'b', 'c'};
  auto const d_data = cudf::detail::make_device_uvector_async(
    h_data, stream, cudf::get_current_device_resource_ref());

  auto const h_input =
    std::vector<string_pair>{{d_data.data(), 1}, {d_data.data() + 1, 1}, {d_data.data() + 2, 1}};
  auto const d_input = cudf::detail::make_device_uvector_async(
    h_input, stream, cudf::get_current_device_resource_ref());
  auto const input = cudf::device_span<string_pair const>{d_input.data(), d_input.size()};
  cudf::make_strings_column(input, stream);
}

TEST_F(StringsFactoryTest, StringBatchConstruction)
{
  auto const stream = cudf::test::get_default_stream();

  auto const h_data = std::vector<char>{'a', 'b', 'c'};
  auto const d_data = cudf::detail::make_device_uvector_async(
    h_data, stream, cudf::get_current_device_resource_ref());

  auto const h_input =
    std::vector<string_pair>{{d_data.data(), 1}, {d_data.data() + 1, 1}, {d_data.data() + 2, 1}};
  auto const d_input = cudf::detail::make_device_uvector_async(
    h_input, stream, cudf::get_current_device_resource_ref());

  std::vector<cudf::device_span<string_pair const>> input(
    10, cudf::device_span<string_pair const>{d_input.data(), d_input.size()});
  cudf::make_strings_column_batch(input, stream);
}
