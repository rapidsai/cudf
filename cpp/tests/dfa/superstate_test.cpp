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

#include <cudf/dfa/superstate.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf_test/base_fixture.hpp>

enum class state : uint8_t { a, b, c, error };
enum class instruction : uint8_t { inc, dec, swap_ac };

inline constexpr state operator+(state lhs, instruction rhs)
{
  switch (rhs) {
    case instruction::inc:
      switch (lhs) {
        case state::a: return state::b;
        case state::b: return state::c;
        case state::c: return state::a;
      }
    case instruction::dec:
      switch (lhs) {
        case state::a: return state::c;
        case state::b: return state::a;
        case state::c: return state::b;
      }
    case instruction::swap_ac:
      switch (lhs) {
        case state::a: return state::c;
        case state::b: return state::b;
        case state::c: return state::a;
      }
  }

  return state::error;
}

using example_superstate = cudf::dfa::detail::superstate<state, instruction, 4>;

struct DfaSuperstateTest : public cudf::test::BaseFixture {
};

TEST_F(DfaSuperstateTest, CanInitializeAllStates)
{
  auto value = example_superstate();

  EXPECT_EQ(value.data(), 0b11100100);
  EXPECT_EQ(value, value);
  EXPECT_EQ(value, example_superstate());
  EXPECT_EQ(value.get(state::a), state::a);
  EXPECT_EQ(value.get(state::b), state::b);
  EXPECT_EQ(value.get(state::c), state::c);
  EXPECT_EQ(value.get(state::error), state::error);
}

TEST_F(DfaSuperstateTest, CanInitializeSpecificValue)
{
  EXPECT_EQ(example_superstate(0b11100100), 0b11100100);
}

TEST_F(DfaSuperstateTest, CanTransitionAllStataes)
{
  auto value = example_superstate();

  value = value + instruction::inc;

  EXPECT_EQ(value.data(), 0b11001001);
  EXPECT_EQ(value.get(state::a), state::b);
  EXPECT_EQ(value.get(state::b), state::c);
  EXPECT_EQ(value.get(state::c), state::a);
  EXPECT_EQ(value.get(state::error), state::error);

  value = value + instruction::swap_ac;

  EXPECT_EQ(value.data(), 0b11100001);
  EXPECT_EQ(value.get(state::a), state::b);
  EXPECT_EQ(value.get(state::b), state::a);
  EXPECT_EQ(value.get(state::c), state::c);
  EXPECT_EQ(value.get(state::error), state::error);

  value = value + instruction::dec;

  EXPECT_EQ(value.data(), 0b11011000);
  EXPECT_EQ(value.get(state::a), state::a);
  EXPECT_EQ(value.get(state::b), state::c);
  EXPECT_EQ(value.get(state::c), state::b);
  EXPECT_EQ(value.get(state::error), state::error);
}

TEST_F(DfaSuperstateTest, CanConcatenateSuperstates)
{
  auto a = example_superstate() + instruction::inc + instruction::swap_ac;
  auto b = example_superstate() + instruction::dec + instruction::swap_ac;
  auto c = example_superstate() + instruction::swap_ac + instruction::inc;

  auto value    = a + b + c;
  auto expected = example_superstate() +                     //
                  instruction::inc + instruction::swap_ac +  //
                  instruction::dec + instruction::swap_ac +  //
                  instruction::swap_ac + instruction::inc;

  EXPECT_EQ(value, expected);
}

CUDF_TEST_PROGRAM_MAIN()
