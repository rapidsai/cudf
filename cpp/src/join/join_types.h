/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#ifndef JOIN_TYPES_H
#define JOIN_TYPES_H

constexpr int JoinNoneValue = -1; /*!< Index value used to signify a mismatch between rows of the left and right table */

/*! Enum class specifying type of join to be performed */
 /*!< Select rows from both tables if they match */
/*!< Select all rows from left table and matched rows from right table */
/*!< Select all rows that match in both tables and those that do not satisfy join condition */
enum class JoinType {
  INNER_JOIN,
  LEFT_JOIN,  
  FULL_JOIN   
};

#endif
