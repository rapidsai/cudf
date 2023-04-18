/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.

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

/**
 * @file list_view.hpp
 * @brief Class definition for cudf::list_view.
 */

namespace cudf {

/**
 * @brief A non-owning, immutable view of device data that represents
 * a list of elements of arbitrary type (including further nested lists).
 */
class list_view {};

}  // namespace cudf
