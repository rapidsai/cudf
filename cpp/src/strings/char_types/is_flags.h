/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>

//
// 8-bit flag for each code-point.
//
constexpr uint8_t IS_DECIMAL(uint8_t x) { return ((x) & (1 << 0)); }
constexpr uint8_t IS_NUMERIC(uint8_t x) { return ((x) & (1 << 1)); }
constexpr uint8_t IS_DIGIT(uint8_t x) { return ((x) & (1 << 2)); }
constexpr uint8_t IS_ALPHA(uint8_t x) { return ((x) & (1 << 3)); }
constexpr uint8_t IS_SPACE(uint8_t x) { return ((x) & (1 << 4)); }
constexpr uint8_t IS_UPPER(uint8_t x) { return ((x) & (1 << 5)); }
constexpr uint8_t IS_LOWER(uint8_t x) { return ((x) & (1 << 6)); }
constexpr uint8_t IS_SPECIAL(uint8_t x) { return ((x) & (1 << 7)); }
constexpr uint8_t IS_ALPHANUM(uint8_t x) { return ((x) & (0x0F)); }
constexpr uint8_t IS_UPPER_OR_LOWER(uint8_t x) { return ((x) & ((1 << 5) | (1 << 6))); }
