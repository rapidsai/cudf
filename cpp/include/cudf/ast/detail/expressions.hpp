/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace cudf {
namespace ast {
namespace detail {
// Forward declaration.
class expression_parser;

/**
 * @brief A generic node that can be evaluated to return a value.
 *
 * This class is a part of a "visitor" pattern with the `expression_parser` class.
 * Nodes inheriting from this class can accept visitors.
 */
struct node {
  virtual cudf::size_type accept(detail::expression_parser& visitor) const = 0;
};

}  // namespace detail

}  // namespace ast

}  // namespace cudf
