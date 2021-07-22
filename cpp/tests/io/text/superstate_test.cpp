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

#include <cudf_test/base_fixture.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/text/superstate.hpp>
#include <cudf/scalar/scalar.hpp>

enum class state : uint8_t { a, b, c, error };
enum class instruction : uint8_t { inc, dec, swap_ac };

inline constexpr state operator+(state const& lhs, instruction const& rhs)
{
  switch (rhs) {
    case instruction::inc:
      switch (lhs) {
        case state::a: return state::b;
        case state::b: return state::c;
        case state::c: return state::a;
        case state::error: return state::error;
      }
    case instruction::dec:
      switch (lhs) {
        case state::a: return state::c;
        case state::b: return state::a;
        case state::c: return state::b;
        case state::error: return state::error;
      }
    case instruction::swap_ac:
      switch (lhs) {
        case state::a: return state::c;
        case state::b: return state::b;
        case state::c: return state::a;
        case state::error: return state::error;
      }
  }

  return state::error;
}

using superstate = cudf::io::text::superstate<4, state>;

struct SuperstateTest : public cudf::test::BaseFixture {
};

TEST_F(SuperstateTest, CanInitializeAllStates)
{
  auto value = superstate();

  EXPECT_EQ(value.data(), 0b11100100);
}

TEST_F(SuperstateTest, CanInitializeSpecificValue)
{
  auto value = superstate(0b01010101);

  EXPECT_EQ(value.data(), 0b01010101);
}

TEST_F(SuperstateTest, CanTransitionExplicitly)
{
  auto value = superstate();

  auto machine = [](state const& lhs, uint8_t const& rhs) {
    return static_cast<state>(static_cast<uint8_t>(lhs) + rhs);
  };

  // this call test the overflow capability of individual states within a superstate. It is
  // possible this becomes UB in the future, in which case this `TEST_F` should be removed.
  value = value.apply(machine, 5);

  EXPECT_EQ(value.data(), 0b00111001);
  EXPECT_EQ(value.get(0), static_cast<state>(1));
}

TEST_F(SuperstateTest, CanTransitionAllStataes)
{
  auto value = superstate();

  value = value + instruction::inc;

  EXPECT_EQ(value.data(), 0b11001001);
  EXPECT_EQ(value.get(0), state::b);

  value = value + instruction::swap_ac;

  EXPECT_EQ(value.data(), 0b11100001);
  EXPECT_EQ(value.get(0), state::b);

  value = value + instruction::dec;

  EXPECT_EQ(value.data(), 0b11011000);
  EXPECT_EQ(value.get(0), state::a);
}

TEST_F(SuperstateTest, CanConcatenateSuperstates)
{
  auto a = superstate() + instruction::inc + instruction::swap_ac;
  auto b = superstate() + instruction::dec + instruction::swap_ac;
  auto c = superstate() + instruction::swap_ac + instruction::inc;

  auto value    = a + b + c;
  auto expected = superstate() +                             //
                  instruction::inc + instruction::swap_ac +  //
                  instruction::dec + instruction::swap_ac +  //
                  instruction::swap_ac + instruction::inc;

  EXPECT_EQ(value.data(), expected.data());
}

CUDF_TEST_PROGRAM_MAIN()
