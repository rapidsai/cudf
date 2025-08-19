/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

// Explicitly disable platform-specific acceleration code
#define DISABLENEON                       1
#define ROARING_DISABLE_X64               1
#define ROARING_DISABLE_AVX               1
#define CROARING_COMPILER_SUPPORTS_AVX512 0

#include <roaring/roaring64.h>