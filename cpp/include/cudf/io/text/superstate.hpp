#pragma once

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {

constexpr unsigned floorlog2(unsigned x) { return x == 1 ? 0 : 1 + floorlog2(x >> 1); }

constexpr unsigned ceillog2(unsigned x) { return x == 1 ? 0 : floorlog2(x - 1) + 1; }

template <uint8_t Bits, typename Enable = void>
struct rep {
};

template <uint8_t Bits>
struct rep<Bits, std::enable_if_t<0 < Bits and Bits <= 8>> {
  using type = uint8_t;
};

template <uint8_t Bits>
struct rep<Bits, std::enable_if_t<8 < Bits and Bits <= 16>> {
  using type = uint16_t;
};

template <uint8_t Bits>
struct rep<Bits, std::enable_if_t<16 < Bits and Bits <= 32>> {
  using type = uint32_t;
};

template <uint8_t Bits>
struct rep<Bits, std::enable_if_t<32 < Bits and Bits <= 64>> {
  using type = uint64_t;
};

template <uint8_t N>
struct superstate_policy {
  static_assert(N > 1 and N <= 16, "superstate supports no more than 16 unique states");
  static constexpr uint8_t BITS = ceillog2(N);
  static constexpr uint8_t MASK = (1 << BITS) - 1;
  using Data                    = typename rep<N * BITS>::type;
};

}  // namespace

namespace cudf {
namespace io {
namespace text {

template <uint8_t N, typename State = uint8_t>
struct superstate {
 public:
  static constexpr uint8_t BITS = superstate_policy<N>::BITS;
  static constexpr uint8_t MASK = superstate_policy<N>::MASK;

  using Data  = typename superstate_policy<N>::Data;
  using Index = uint8_t;

 private:
  Data _data;

 public:
  /**
   * @brief creates a superstate which represents all possible states and
   * applied transitions
   */
  constexpr superstate() : _data(0)
  {
    for (auto i = 0; i < N; i++) { _data |= static_cast<Data>(i) << (i * BITS); }
  }

  explicit inline constexpr superstate(Data data) : _data(data) {}

  inline constexpr Data data() const { return _data; }

  explicit inline constexpr operator State() const { return static_cast<State>(_data & MASK); }

  inline constexpr State get(Index idx) const
  {
    return static_cast<State>((_data >> idx * BITS) & MASK);
  }

  inline constexpr void set(Index idx, State state)
  {
    // removing `& MASK` here may result in less instructions, but will result in UB. This may
    // be a fine trade-off, as integer-overflow was never an intended use case.
    _data |= (static_cast<Data>(state) & MASK) << idx * BITS;
  }

  inline constexpr void reset(Index idx, State state)
  {
    _data &= ~(MASK << idx * BITS);
    _data |= static_cast<Data>(state) << idx * BITS;
  }

  template <typename BinaryOp, typename RHS>
  inline constexpr superstate apply(BinaryOp const& op, RHS const& rhs)
  {
    superstate<N, State> result(0);
    for (uint8_t pre = 0; pre < N; pre++) {
      auto const mid  = get(pre);
      auto const post = op(mid, rhs);
      result.set(pre, post);
    }
    return result;
  }

  template <typename BinaryOp>
  inline constexpr superstate apply(BinaryOp const& op)
  {
    superstate<N, State> result(0);
    for (uint8_t pre = 0; pre < N; pre++) {
      auto const mid  = get(pre);
      auto const post = op(mid);
      result.set(pre, post);
    }
    return result;
  }
};

template <typename State, uint8_t N, typename Instruction>
inline constexpr superstate<N, State> operator+(superstate<N, State> lhs, Instruction rhs)
{
  return lhs.apply([&](State state) { return state + rhs; });
}

template <typename State, uint8_t N>
inline constexpr superstate<N, State> operator+(superstate<N, State> lhs, superstate<N, State> rhs)
{
  using Index = typename superstate<N, State>::Index;
  return lhs.apply([&](State state) { return rhs.get(static_cast<Index>(state)); });
}

}  // namespace text
}  // namespace io
}  // namespace cudf
