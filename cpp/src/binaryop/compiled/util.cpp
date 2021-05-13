/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "traits.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

namespace cudf::binops::compiled {

namespace {
// common_type
struct common_type_functor {
  template <typename TypeLhs, typename TypeRhs>
  struct nested_common_type_functor {
    template <typename TypeOut>
    data_type operator()()
    {
      // If common_type exists
      if constexpr (cudf::has_common_type_v<TypeOut, TypeLhs, TypeRhs>) {
        using TypeCommon = typename std::common_type<TypeOut, TypeLhs, TypeRhs>::type;
        return data_type{type_to_id<TypeCommon>()};
      } else if constexpr (cudf::has_common_type_v<TypeLhs, TypeRhs>) {
        using TypeCommon = typename std::common_type<TypeLhs, TypeRhs>::type;
        // Eg. d=t-t
        return data_type{type_to_id<TypeCommon>()};
      }
      return data_type{type_id::EMPTY};
    }
  };
  template <typename TypeLhs, typename TypeRhs>
  data_type operator()(data_type out)
  {
    return type_dispatcher(out, nested_common_type_functor<TypeLhs, TypeRhs>{});
  }
};
}  // namespace

data_type get_common_type(data_type out, data_type lhs, data_type rhs)
{
  return double_type_dispatcher(lhs, rhs, common_type_functor{}, out);
}

}  // namespace cudf::binops::compiled
