/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#define SHIFT_FOR_NEW_CP 0
#define NEW_CP_MASK 0x1fffff

#define BYTES_LESS_1_SHIFT 21
#define BYTES_LESS_1_MASK 0x3

#define MULTICHAR_SHIFT 23
#define MULTICHAR_MASK 1

#define TOKEN_CAT_SHIFT 24
#define TOKEN_CAT_MASK 7
#define TOKEN_CAT_ADD_SPACE 0
#define TOKEN_CAT_ADD_SPACE_IF_LOWER 1
#define TOKEN_CAT_REMOVE_CHAR 2
#define TOKEN_CAT_REMOVE_CHAR_IF_LOWER 3
#define TOKEN_CAT_ALWAYS_REPLACE 4

#define SPACE_CODE_POINT 32
#define MAX_NEW_CHARS 3
