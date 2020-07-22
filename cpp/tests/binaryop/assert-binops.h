/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
 *
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
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

#include <tests/binaryop/util/operation.h>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/cudf_gtest.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/traits.hpp>

#include <limits>

namespace cudf {
namespace test {
namespace binop {

// This is used to convert the expected binop result computed by the test utilities and the
// result returned by the binop operation into string, which is then used for display purposes
// when the values do not match.
struct stringify_out_values {
  template <typename TypeOut, std::enable_if_t<!is_chrono<TypeOut>()>* = nullptr>
  std::string operator()(TypeOut lhs, TypeOut rhs) const
  {
    std::stringstream out_str;
    out_str << "lhs: " << lhs << "\nrhs: " << rhs;
    return out_str.str();
  }

  template <typename TypeOut, std::enable_if_t<is_timestamp<TypeOut>()>* = nullptr>
  std::string operator()(TypeOut lhs, TypeOut rhs) const
  {
    std::stringstream out_str;
    out_str << "lhs: " << lhs.time_since_epoch().count()
            << "\nrhs: " << rhs.time_since_epoch().count();
    return out_str.str();
  }

  template <typename TypeOut, std::enable_if_t<is_duration<TypeOut>()>* = nullptr>
  std::string operator()(TypeOut lhs, TypeOut rhs) const
  {
    std::stringstream out_str;
    out_str << "lhs: " << lhs.count() << "\nrhs: " << rhs.count();
    return out_str.str();
  }
};

// This comparator can be used to compare two values that are within a max ULP error.
// This is typically used to compare floating point values computed on CPU and GPU which is
// expected to be *near* equal, or when computing large numbers can yield ULP errors
template <typename TypeOut>
struct NearEqualComparator {
  double ulp_;

  NearEqualComparator(double ulp) : ulp_(ulp) {}

  bool operator()(TypeOut const& lhs, TypeOut const& rhs) const
  {
    return (std::fabs(lhs - rhs) <=
              std::numeric_limits<TypeOut>::epsilon() * std::fabs(lhs + rhs) * ulp_ ||
            std::fabs(lhs - rhs) < std::numeric_limits<TypeOut>::min());
  }
};

template <typename TypeOut,
          typename TypeLhs,
          typename TypeRhs,
          typename TypeOp,
          typename ValueComparator = std::equal_to<TypeOut>,
          typename ScalarType      = cudf::scalar_type_t<TypeLhs>>
void ASSERT_BINOP(column_view const& out,
                  scalar const& lhs,
                  column_view const& rhs,
                  TypeOp&& op,
                  ValueComparator const& value_comparator = ValueComparator())
{
  auto lhs_h    = static_cast<ScalarType const&>(lhs).operator TypeLhs();
  auto rhs_h    = cudf::test::to_host<TypeRhs>(rhs);
  auto rhs_data = rhs_h.first;
  auto out_h    = cudf::test::to_host<TypeOut>(out);
  auto out_data = out_h.first;

  ASSERT_EQ(out_data.size(), rhs_data.size());
  for (size_t i = 0; i < out_data.size(); ++i) {
    auto lhs = out_data[i];
    auto rhs = (TypeOut)(op(lhs_h, rhs_data[i]));
    ASSERT_TRUE(value_comparator(lhs, rhs)) << stringify_out_values{}(lhs, rhs);
  }

  if (rhs.nullable()) {
    ASSERT_TRUE(out.nullable());
    auto rhs_valid = rhs_h.second;
    auto out_valid = out_h.second;

    uint32_t lhs_valid = (lhs.is_valid() ? std::numeric_limits<bitmask_type>::max() : 0);
    ASSERT_EQ(out_valid.size(), rhs_valid.size());
    for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
      ASSERT_EQ(out_valid[i], (lhs_valid & rhs_valid[i]));
    }
  } else {
    if (lhs.is_valid()) {
      ASSERT_FALSE(out.nullable());
    } else {
      auto out_valid = out_h.second;
      for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
        ASSERT_EQ(out_valid[i], 0);
      }
    }
  }
}  // namespace binop

template <typename TypeOut,
          typename TypeLhs,
          typename TypeRhs,
          typename TypeOp,
          typename ValueComparator = std::equal_to<TypeOut>,
          typename ScalarType      = cudf::scalar_type_t<TypeRhs>>
void ASSERT_BINOP(column_view const& out,
                  column_view const& lhs,
                  scalar const& rhs,
                  TypeOp&& op,
                  ValueComparator const& value_comparator = ValueComparator())
{
  auto rhs_h    = static_cast<ScalarType const&>(rhs).operator TypeRhs();
  auto lhs_h    = cudf::test::to_host<TypeLhs>(lhs);
  auto lhs_data = lhs_h.first;
  auto out_h    = cudf::test::to_host<TypeOut>(out);
  auto out_data = out_h.first;

  ASSERT_EQ(out_data.size(), lhs_data.size());
  for (size_t i = 0; i < out_data.size(); ++i) {
    auto lhs = out_data[i];
    auto rhs = (TypeOut)(op(lhs_data[i], rhs_h));
    ASSERT_TRUE(value_comparator(lhs, rhs)) << stringify_out_values{}(lhs, rhs);
  }

  if (lhs.nullable()) {
    ASSERT_TRUE(out.nullable());
    auto lhs_valid = lhs_h.second;
    auto out_valid = out_h.second;

    uint32_t rhs_valid = (rhs.is_valid() ? std::numeric_limits<bitmask_type>::max() : 0);
    ASSERT_EQ(out_valid.size(), lhs_valid.size());
    for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
      ASSERT_EQ(out_valid[i], (rhs_valid & lhs_valid[i]));
    }
  } else {
    if (rhs.is_valid()) {
      ASSERT_FALSE(out.nullable());
    } else {
      auto out_valid = out_h.second;
      for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
        ASSERT_EQ(out_valid[i], 0);
      }
    }
  }
}  // namespace test

template <typename TypeOut,
          typename TypeLhs,
          typename TypeRhs,
          typename TypeOp,
          typename ValueComparator = std::equal_to<TypeOut>>
void ASSERT_BINOP(column_view const& out,
                  column_view const& lhs,
                  column_view const& rhs,
                  TypeOp&& op,
                  ValueComparator const& value_comparator = ValueComparator())
{
  auto lhs_h    = cudf::test::to_host<TypeLhs>(lhs);
  auto lhs_data = lhs_h.first;
  auto rhs_h    = cudf::test::to_host<TypeRhs>(rhs);
  auto rhs_data = rhs_h.first;
  auto out_h    = cudf::test::to_host<TypeOut>(out);
  auto out_data = out_h.first;

  ASSERT_EQ(out_data.size(), lhs_data.size());
  ASSERT_EQ(out_data.size(), rhs_data.size());
  for (size_t i = 0; i < out_data.size(); ++i) {
    auto lhs = out_data[i];
    auto rhs = (TypeOut)(op(lhs_data[i], rhs_data[i]));
    ASSERT_TRUE(value_comparator(lhs, rhs)) << stringify_out_values{}(lhs, rhs);
  }

  if (lhs.nullable() and rhs.nullable()) {
    ASSERT_TRUE(out.nullable());
    auto lhs_valid = lhs_h.second;
    auto rhs_valid = rhs_h.second;
    auto out_valid = out_h.second;

    ASSERT_EQ(out_valid.size(), lhs_valid.size());
    ASSERT_EQ(out_valid.size(), rhs_valid.size());
    for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
      ASSERT_EQ(out_valid[i], (lhs_valid[i] & rhs_valid[i]));
    }
  } else if (not lhs.nullable() and rhs.nullable()) {
    ASSERT_TRUE(out.nullable());
    auto rhs_valid = rhs_h.second;
    auto out_valid = out_h.second;

    ASSERT_EQ(out_valid.size(), rhs_valid.size());
    for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
      ASSERT_EQ(out_valid[i], rhs_valid[i]);
    }
  } else if (lhs.nullable() and not rhs.nullable()) {
    ASSERT_TRUE(out.nullable());
    auto lhs_valid = lhs_h.second;
    auto out_valid = out_h.second;

    ASSERT_EQ(out_valid.size(), lhs_valid.size());
    for (size_type i = 0; i < num_bitmask_words(out_data.size()); ++i) {
      ASSERT_EQ(out_valid[i], lhs_valid[i]);
    }
  } else {
    ASSERT_FALSE(out.nullable());
  }
}

}  // namespace binop
}  // namespace test
}  // namespace cudf
