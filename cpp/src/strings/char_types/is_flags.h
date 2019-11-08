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
// Flags for each character are defined in char_flags.h
//
#define IS_DECIMAL(x)  (x &  1)
#define IS_NUMERIC(x)  (x &  2)
#define IS_DIGIT(x)    (x &  4)
#define IS_ALPHA(x)    (x &  8)
#define IS_ALPHANUM(x) (x & 15)
#define IS_SPACE(x)    (x & 16)
#define IS_UPPER(x)    (x & 32)
#define IS_LOWER(x)    (x & 64)
