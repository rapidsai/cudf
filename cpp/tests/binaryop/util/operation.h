/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cmath>
#include <cstdint>
#include <cudf/utilities/traits.hpp>
#include <type_traits>

namespace cudf {
namespace library {
namespace operation {

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Add {
  // Allow sum between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename OutT                        = TypeOut,
            typename std::enable_if<cudf::is_chrono<OutT>() && cudf::is_chrono<TypeLhs>() &&
                                      cudf::is_chrono<TypeRhs>(),
                                    void>::type* = nullptr>
  OutT operator()(TypeLhs lhs, TypeRhs rhs) const
  {
    return lhs + rhs;
  }

  template <typename OutT                        = TypeOut,
            typename std::enable_if<!cudf::is_chrono<OutT>() || !cudf::is_chrono<TypeLhs>() ||
                                      !cudf::is_chrono<TypeRhs>(),
                                    void>::type* = nullptr>
  OutT operator()(TypeLhs lhs, TypeRhs rhs) const
  {
    using TypeCommon = typename std::common_type<OutT, TypeLhs, TypeRhs>::type;
    return static_cast<OutT>(static_cast<TypeCommon>(lhs) + static_cast<TypeCommon>(rhs));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Sub {
  // Allow difference between chronos only when both input and output types
  // are chronos. Unsupported combinations will fail to compile
  template <typename OutT                        = TypeOut,
            typename std::enable_if<cudf::is_chrono<OutT>() && cudf::is_chrono<TypeLhs>() &&
                                      cudf::is_chrono<TypeRhs>(),
                                    void>::type* = nullptr>
  OutT operator()(TypeLhs lhs, TypeRhs rhs) const
  {
    return lhs - rhs;
  }

  template <typename OutT                        = TypeOut,
            typename std::enable_if<!cudf::is_chrono<OutT>() || !cudf::is_chrono<TypeLhs>() ||
                                      !cudf::is_chrono<TypeRhs>(),
                                    void>::type* = nullptr>
  OutT operator()(TypeLhs lhs, TypeRhs rhs) const
  {
    using TypeCommon = typename std::common_type<OutT, TypeLhs, TypeRhs>::type;
    return static_cast<OutT>(static_cast<TypeCommon>(lhs) - static_cast<TypeCommon>(rhs));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Mul {
  template <typename OutT                                                           = TypeOut,
            typename std::enable_if<!cudf::is_duration_t<OutT>::value, void>::type* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) const
  {
    using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
    return static_cast<TypeOut>(static_cast<TypeCommon>(lhs) * static_cast<TypeCommon>(rhs));
  }

  template <typename OutT                                                          = TypeOut,
            typename std::enable_if<cudf::is_duration_t<OutT>::value, void>::type* = nullptr>
  TypeOut operator()(TypeLhs x, TypeRhs y) const
  {
    return DurationProduct<TypeOut>(x, y);
  }

  template <
    typename OutT,
    typename LhsT,
    typename RhsT,
    typename std::enable_if<(cudf::is_duration_t<LhsT>::value && std::is_integral<RhsT>::value) ||
                              (cudf::is_duration_t<RhsT>::value && std::is_integral<LhsT>::value),
                            void>::type* = nullptr>
  OutT DurationProduct(LhsT x, RhsT y) const
  {
    return x * y;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Div {
  template <typename LhsT                                                           = TypeLhs,
            typename std::enable_if<!cudf::is_duration_t<LhsT>::value, void>::type* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
    return static_cast<TypeOut>(static_cast<TypeCommon>(lhs) / static_cast<TypeCommon>(rhs));
  }

  template <typename LhsT                                                          = TypeLhs,
            typename std::enable_if<cudf::is_duration_t<LhsT>::value, void>::type* = nullptr>
  TypeOut operator()(TypeLhs x, TypeRhs y) const
  {
    return DurationDivide<TypeOut>(x, y);
  }

  template <
    typename OutT,
    typename LhsT,
    typename RhsT,
    typename std::enable_if<(std::is_integral<RhsT>::value || cudf::is_duration_t<RhsT>::value),
                            void>::type* = nullptr>
  OutT DurationDivide(LhsT x, RhsT y) const
  {
    return x / y;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct TrueDiv {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(static_cast<double>(lhs) / static_cast<double>(rhs));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct FloorDiv {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(floor(static_cast<double>(lhs) / static_cast<double>(rhs)));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Mod {
  template <
    typename OutT = TypeOut,
    typename LhsT = TypeLhs,
    typename RhsT = TypeRhs,
    std::enable_if_t<
      (std::is_integral<typename std::common_type<OutT, LhsT, RhsT>::type>::value)>* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
    return static_cast<TypeOut>(static_cast<TypeCommon>(lhs) % static_cast<TypeCommon>(rhs));
  }

  template <typename OutT                                   = TypeOut,
            typename LhsT                                   = TypeLhs,
            typename RhsT                                   = TypeRhs,
            std::enable_if_t<(std::is_same<typename std::common_type<OutT, LhsT, RhsT>::type,
                                           float>::value)>* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(fmod(static_cast<float>(lhs), static_cast<float>(rhs)));
  }

  template <typename OutT                                    = TypeOut,
            typename LhsT                                    = TypeLhs,
            typename RhsT                                    = TypeRhs,
            std::enable_if_t<(std::is_same<typename std::common_type<OutT, LhsT, RhsT>::type,
                                           double>::value)>* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(fmod(static_cast<double>(lhs), static_cast<double>(rhs)));
  }

  // Mod with duration types - duration % (integral or a duration) = duration
  template <typename LhsT                                                = TypeLhs,
            typename OutT                                                = TypeOut,
            typename std::enable_if_t<cudf::is_duration_t<LhsT>::value &&
                                      cudf::is_duration_t<OutT>::value>* = nullptr>
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return lhs % rhs;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Pow {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(pow(static_cast<double>(lhs), static_cast<double>(rhs)));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Equal {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x == y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct NotEqual {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x != y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Less {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x < y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct Greater {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x > y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct LessEqual {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x <= y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct GreaterEqual {
  TypeOut operator()(TypeLhs x, TypeRhs y) { return (x >= y); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct BitwiseAnd {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return (lhs & rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct BitwiseOr {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return (lhs | rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct BitwiseXor {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return (lhs ^ rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct LogicalAnd {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return static_cast<TypeOut>(lhs && rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct LogicalOr {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return static_cast<TypeOut>(lhs || rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct ShiftLeft {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return static_cast<TypeOut>(lhs << rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct ShiftRight {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs) { return static_cast<TypeOut>(lhs >> rhs); }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct ShiftRightUnsigned {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(static_cast<std::make_unsigned_t<TypeLhs>>(lhs) >> rhs);
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct LogBase {
  TypeOut operator()(TypeLhs lhs, TypeRhs rhs)
  {
    return static_cast<TypeOut>(std::log(static_cast<double>(lhs)) /
                                std::log(static_cast<double>(rhs)));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct PMod {
  using CommonArgsT = typename std::common_type<TypeLhs, TypeRhs>::type;

  TypeOut operator()(TypeLhs x, TypeRhs y) const
  {
    CommonArgsT xconv{static_cast<CommonArgsT>(x)};
    CommonArgsT yconv{static_cast<CommonArgsT>(y)};
    auto rem = std::fmod(xconv, yconv);
    if (rem < 0) rem = std::fmod(rem + yconv, yconv);
    return static_cast<TypeOut>(rem);
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct ATan2 {
  TypeOut operator()(TypeLhs x, TypeRhs y) const
  {
    return static_cast<TypeOut>(std::atan2(static_cast<double>(x), static_cast<double>(y)));
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct PyMod {
  TypeOut operator()(TypeLhs x, TypeRhs y) const
  {
    if constexpr (std::is_floating_point_v<TypeLhs> or std::is_floating_point_v<TypeRhs>) {
      double x1 = static_cast<double>(x);
      double y1 = static_cast<double>(y);
      return fmod(fmod(x1, y1) + y1, y1);
    } else {
      return ((x % y) + y) % y;
    }
    return {};
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct NullEquals {
  TypeOut operator()(TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) const
  {
    output_valid = true;
    if (!lhs_valid && !rhs_valid) return true;
    using common_t = std::common_type_t<TypeLhs, TypeRhs>;
    if (lhs_valid && rhs_valid) return static_cast<common_t>(x) == static_cast<common_t>(y);
    return false;
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct NullMax {
  TypeOut operator()(TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) const
  {
    output_valid = lhs_valid or rhs_valid;
    if (lhs_valid or rhs_valid) {
      return (lhs_valid and (!rhs_valid or static_cast<TypeOut>(x) > static_cast<TypeOut>(y)))
               ? static_cast<TypeOut>(x)
               : static_cast<TypeOut>(y);
    } else
      return TypeOut{};
  }
};

template <typename TypeOut, typename TypeLhs, typename TypeRhs>
struct NullMin {
  TypeOut operator()(TypeLhs x, TypeRhs y, bool lhs_valid, bool rhs_valid, bool& output_valid) const
  {
    output_valid = lhs_valid or rhs_valid;
    if (lhs_valid or rhs_valid) {
      return (lhs_valid and (!rhs_valid or static_cast<TypeOut>(x) < static_cast<TypeOut>(y)))
               ? static_cast<TypeOut>(x)
               : static_cast<TypeOut>(y);
    } else
      return TypeOut{};
  }
};

}  // namespace operation
}  // namespace library
}  // namespace cudf
