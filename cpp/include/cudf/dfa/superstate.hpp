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

#pragma once

#include <cstdint>
#include <type_traits>

namespace cudf {
namespace dfa {
namespace detail {

namespace {

/**
 * @brief specialization helper class for superstate used to minimize storage requirement.
 *
 * @note provides no specialization for:
 * N =  0 - not a state
 * N =  1 - better represented by the underlying type
 * N > 16 - better represented as multiple indivual states
 *
 * @tparam N number of states the superstate tracks
 */
template <uint8_t N, typename Enable = void>
struct superstate_policy {
};

template <uint8_t N>
struct superstate_policy<N, typename std::enable_if<2 <= N and N <= 4>::type> {
  static const uint8_t BITS = 2;
  static const uint8_t MASK = 0b11;

  using Data = uint8_t;
};

template <uint8_t N>
struct superstate_policy<N, typename std::enable_if<5 <= N and N <= 5>::type> {
  static const uint8_t BITS = 3;
  static const uint8_t MASK = 0b111;

  using Data = uint16_t;
};

template <uint8_t N>
struct superstate_policy<N, typename std::enable_if<6 <= N and N <= 8>::type> {
  static const uint8_t BITS = 3;
  static const uint8_t MASK = 0b111;

  using Data = uint32_t;
};

template <uint8_t N>
struct superstate_policy<N, typename std::enable_if<9 <= N and N <= 16>::type> {
  static const uint8_t BITS = 4;
  static const uint8_t MASK = 0b1111;

  using Data = uint64_t;
};

}  // namespace

/**
 * @brief Represents all possible dfa states and the transitions applied to them
 *
 * Used to compute dfa states and transitions in parallel. Stores states in a compressed format.
 *
 * Requires `N * log2(N)` storage bits:
 * - up to  4 states + all transitions in 1 byte  (uint8_t)
 * - up to  5 states + all transtiions in 2 bytes (uint16_t)
 * - up to  8 states + all transitions in 4 bytes (uint32_t)
 * - up to 16 states + all transitions in 8 bytes (uint64_t)
 *
 * ```
 * // alias superstate for convienence.
 * // csv_state has 7 unique states, and will therefore require a total of 4 bytes (uint32_t)
 * using csv_superstate = superstate<csv_state, csv_token, 7>;
 *
 * // apply state transitions in "parallel"
 * auto a = csv_superstate() + csv_token::comment + csv_token::other + csv_token::newline;
 * auto b = csv_superstate() + csv_token::other + csv_token::delimiter;
 * auto c = csv_superstate() + csv_token::other + csv_token::delimiter;
 *
 * // given record_end as the initial state, get the resulting state after all transitions.
 * auto final = a + b + c;
 * auto result_a = final.get(csv_state::record_end);   // csv_state::field_end
 * auto result_b = final.get(csv_state::quoted_field); // csv_state::quoted_field
 * auto result_c = final.get(csv_state::field_end);    // csv_state::invalid
 * ```
 *
 * @note this class is constexpr, and can be used on both host and device
 *
 * @tparam State state which will be tracked - up to 16 unique values. usually an enum class.
 * @tparam Instruction type which can be added to a state to transition in to a new state.
 * @tparam N total number of states representable by State, up to 16 unique states.
 */
template <typename State, typename Instruction, uint8_t N>
struct superstate {
 private:
  static const uint8_t BITS = superstate_policy<N>::BITS;
  static const uint8_t MASK = superstate_policy<N>::MASK;

  using Data = typename superstate_policy<N>::Data;

  Data _data;

 public:
  /**
   * @brief creates a superstate which represents all possible states and applied transitions
   */
  inline constexpr superstate() : _data(0)
  {
    for (auto i = 0; i < N; i++) { _data |= i << (i * BITS); }
  }

  /**
   * @brief creates a superstate which represents only a single state
   */
  inline constexpr superstate(State state) : _data(0)
  {
    for (auto i = 0; i < N; i++) { _data |= static_cast<Data>(state) << (i * BITS); }
  }

  /**
   * @brief create a superstate with a specific underlying value
   * @param data the underlying value
   */
  inline constexpr superstate(Data data) : _data(data) {}

  inline constexpr Data data() const { return _data; }

  /**
   * @brief converts to a single state using the least significant bits of the underlying data
   *
   * @return State a state created from the least significant bits of the underlying data
   */
  inline constexpr operator State() const { return static_cast<State>(_data & MASK); }

  /**
   * @brief transitions each underlying state using the provided instruction
   *
   * @note the + operator is called N times, once for each state.
   *
   * @param rhs the instruction used to transition all states
   * @return the result of all transitions
   */
  inline constexpr superstate operator+(Instruction rhs) const
  {
    superstate result(0);
    for (auto i = 0; i < N; i++) { result.set<false>(i, get(i) + rhs); }
    return result;
  }

  /**
   * @brief concatenates all underlying states and transitions in to a new superstate
   *
   * @note this operator is associative (but non-commutative) and can be used in prefix sums / scans
   *
   * @param rhs a superstate representing a series of transitions
   * @return constexpr superstate
   */
  inline constexpr superstate operator+(superstate rhs) const
  {
    superstate result(0);
    for (auto i = 0; i < N; i++) { result.set<false>(i, rhs.get(get(i))); }
    return result;
  }

  /**
   * @brief compares the underlying data equality. all represented states must match exactly.
   *
   * @param rhs value to be compared
   * @return true if all represented states match
   * @return false if at least one represented state does not match
   */
  inline constexpr bool operator==(superstate rhs) const { return _data == rhs._data; }

  /**
   * @brief
   *
   * @param rhs value to be compared
   * @return false if all represented states match
   * @return true if at least one represented state does not match
   */
  inline constexpr bool operator!=(superstate rhs) const { return _data != rhs._data; }

  /**
   * @brief get the final state after all represented transitions have been applied to the input
   *
   * @param state starting state before any transitions
   * @return State ending state after all transitions
   */
  inline constexpr State get(State state) const { return get(static_cast<uint8_t>(state)); }

 private:
  inline constexpr State get(uint8_t n) const
  {
    return static_cast<State>((_data >> n * BITS) & MASK);
  }

  template <bool reset = true>
  inline constexpr void set(uint8_t n, State state)
  {
    if (reset) { _data &= ~(MASK << n * BITS); }
    _data |= static_cast<Data>(state) << n * BITS;
  }
};

}  // namespace detail
}  // namespace dfa
}  // namespace cudf
