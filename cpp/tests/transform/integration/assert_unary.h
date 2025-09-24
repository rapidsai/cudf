/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>

namespace transformation {
template <typename TypeOut, typename TypeIn, typename TypeOpe>
void ASSERT_UNARY(cudf::column_view const& out, cudf::column_view const& in, TypeOpe&& ope)
{
  auto in_h     = cudf::test::to_host<TypeIn>(in);
  auto in_data  = in_h.first;
  auto out_h    = cudf::test::to_host<TypeOut>(out);
  auto out_data = out_h.first;

  ASSERT_TRUE(out_data.size() == in_data.size());

  auto begin = thrust::make_zip_iterator(cuda::std::make_tuple(in_data.begin(), out_data.begin()));
  auto end   = thrust::make_zip_iterator(cuda::std::make_tuple(in_data.end(), out_data.end()));

  std::for_each(begin, end, [ope](auto const& zipped) {
    auto [in_val, out_val] = zipped;
    EXPECT_EQ(out_val, static_cast<TypeOut>(ope(in_val)));
  });

  auto in_valid  = in_h.second;
  auto out_valid = out_h.second;

  ASSERT_TRUE(out_valid.size() == in_valid.size());

  auto valid_begin =
    thrust::make_zip_iterator(cuda::std::make_tuple(in_valid.begin(), out_valid.begin()));
  auto valid_end =
    thrust::make_zip_iterator(cuda::std::make_tuple(in_valid.end(), out_valid.end()));

  std::for_each(valid_begin, valid_end, [](auto const& zipped) {
    auto [in_flag, out_flag] = zipped;
    EXPECT_EQ(out_flag, in_flag);
  });
}

}  // namespace transformation
