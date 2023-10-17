/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

namespace transformation {
template <typename TypeOut, typename TypeIn, typename TypeOpe>
void ASSERT_UNARY(cudf::column_view const& out, cudf::column_view const& in, TypeOpe&& ope)
{
  auto in_h     = cudf::test::to_host<TypeIn>(in);
  auto in_data  = in_h.first;
  auto out_h    = cudf::test::to_host<TypeOut>(out);
  auto out_data = out_h.first;

  ASSERT_TRUE(out_data.size() == in_data.size());

  auto data_comparator = [ope](TypeIn const& in, TypeOut const& out) {
    EXPECT_EQ(out, static_cast<TypeOut>(ope(in)));
    return true;
  };
  std::equal(in_data.begin(), in_data.end(), out_data.begin(), data_comparator);

  auto in_valid  = in_h.second;
  auto out_valid = out_h.second;

  ASSERT_TRUE(out_valid.size() == in_valid.size());
  auto valid_comparator = [](bool const& in, bool const& out) {
    EXPECT_EQ(out, in);
    return true;
  };
  std::equal(in_valid.begin(), in_valid.end(), out_valid.begin(), valid_comparator);
}

}  // namespace transformation
