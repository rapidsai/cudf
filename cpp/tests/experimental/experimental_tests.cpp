/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <tests/utilities/GTest.hpp>
#include <tests/utilities/TypeList.hpp>

#include <gmock/gmock.h>

#include <cxxabi.h>
#include <iostream>
#include <typeinfo>

using namespace cudf::test;

template <typename T>
class ExperiementalTest : public ::testing::Test {};

using TestTypes = CrossJoin<Types<int, float>, Types<char, double> >;

TYPED_TEST_CASE(ExperiementalTest, TestTypes);

template <typename T>
std::string type_name() {
  int status;
  char *realname;
  realname = abi::__cxa_demangle(typeid(T).name(), 0, 0, &status);
  std::string name{realname};
  free(realname);
  return name;
}

TYPED_TEST(ExperiementalTest, PrintTypes) {
  std::cout << "typename: " << type_name<TypeParam>() << std::endl;
}
