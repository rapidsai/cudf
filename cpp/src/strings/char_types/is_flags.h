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

//
// 8-bit flag for each code-point.
//
#define IS_DECIMAL(x)  ((x) & (1 << 0))
#define IS_NUMERIC(x)  ((x) & (1 << 1))
#define IS_DIGIT(x)    ((x) & (1 << 2))
#define IS_ALPHA(x)    ((x) & (1 << 3))
#define IS_SPACE(x)    ((x) & (1 << 4))
#define IS_UPPER(x)    ((x) & (1 << 5))
#define IS_LOWER(x)    ((x) & (1 << 6))
#define IS_ALPHANUM(x) ((x) & (0x0F))
