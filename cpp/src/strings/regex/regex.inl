/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <strings/char_types/is_flags.h>
#include <strings/utf8.cuh>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/string_view.cuh>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief This holds the state information when evaluating a string
 * against a regex pattern.
 *
 * There are 2 instances of this per string managed in the reljunk class.
 * As each regex instruction is evaluated for a string, the result is
 * reflected here. The regexec function updates and manages this state data.
 */
struct alignas(8) relist {
  /**
   * @brief Compute the memory size for the state data.
   */
  constexpr inline static std::size_t data_size_for(int32_t insts)
  {
    return ((sizeof(ranges[0]) + sizeof(inst_ids[0])) * insts) +
           cudf::util::div_rounding_up_unsafe(insts, 8);
  }

  /**
   * @brief Compute the aligned memory allocation size.
   */
  constexpr inline static std::size_t alloc_size(int32_t insts)
  {
    return cudf::util::round_up_unsafe<size_t>(data_size_for(insts) + sizeof(relist),
                                               sizeof(ranges[0]));
  }

  struct alignas(16) restate {
    int2 range;
    int32_t inst_id;
    int32_t reserved;
  };

  __device__ __forceinline__ relist(int16_t insts, u_char* data = nullptr)
    : masksize(cudf::util::div_rounding_up_unsafe(insts, 8))
  {
    auto ptr = data == nullptr ? reinterpret_cast<u_char*>(this) + sizeof(relist) : data;
    ranges   = reinterpret_cast<int2*>(ptr);
    ptr += insts * sizeof(ranges[0]);
    inst_ids = reinterpret_cast<int16_t*>(ptr);
    ptr += insts * sizeof(inst_ids[0]);
    mask = ptr;
    reset();
  }

  __device__ __forceinline__ void reset()
  {
    memset(mask, 0, masksize);
    size = 0;
  }

  __device__ __forceinline__ bool activate(int32_t id, int32_t begin, int32_t end)
  {
    if (readMask(id)) { return false; }
    writeMask(id);
    inst_ids[size] = static_cast<int16_t>(id);
    ranges[size]   = int2{begin, end};
    ++size;
    return true;
  }

  __device__ __forceinline__ restate get_state(int16_t idx) const
  {
    return restate{ranges[idx], inst_ids[idx]};
  }

  __device__ __forceinline__ int16_t get_size() const { return size; }

 private:
  int16_t size{};
  int16_t const masksize;
  int32_t reserved;
  int2* __restrict__ ranges;       // pair per instruction
  int16_t* __restrict__ inst_ids;  // one per instruction
  u_char* __restrict__ mask;       // bit per instruction

  __device__ __forceinline__ void writeMask(int32_t pos) const
  {
    u_char const uc = 1 << (pos & 7);
    mask[pos >> 3] |= uc;
  }

  __device__ __forceinline__ bool readMask(int32_t pos) const
  {
    u_char const uc = mask[pos >> 3];
    return static_cast<bool>((uc >> (pos & 7)) & 1);
  }
};

__device__ __forceinline__ reprog_device::reljunk::reljunk(relist* list1,
                                                           relist* list2,
                                                           reinst const inst)
  : list1(list1), list2(list2)
{
  if (inst.type == CHAR || inst.type == BOL) {
    starttype = inst.type;
    startchar = inst.u1.c;
  }
}

__device__ __forceinline__ void reprog_device::reljunk::swaplist()
{
  auto tmp = list1;
  list1    = list2;
  list2    = tmp;
}

/**
 * @brief Utility to check a specific character against this class instance.
 *
 * @param ch A 4-byte UTF-8 character.
 * @param codepoint_flags Used for mapping a character to type for builtin classes.
 * @return true if the character matches
 */
__device__ __forceinline__ bool reclass_device::is_match(char32_t const ch,
                                                         uint8_t const* codepoint_flags) const
{
  for (int i = 0; i < count; ++i) {
    if ((ch >= literals[i * 2]) && (ch <= literals[(i * 2) + 1])) { return true; }
  }

  if (!builtins) return false;
  uint32_t codept = utf8_to_codepoint(ch);
  if (codept > 0x00FFFF) return false;
  int8_t fl = codepoint_flags[codept];
  if ((builtins & 1) && ((ch == '_') || IS_ALPHANUM(fl)))  // \w
    return true;
  if ((builtins & 2) && IS_SPACE(fl))  // \s
    return true;
  if ((builtins & 4) && IS_DIGIT(fl))  // \d
    return true;
  if ((builtins & 8) && ((ch != '\n') && (ch != '_') && !IS_ALPHANUM(fl)))  // \W
    return true;
  if ((builtins & 16) && !IS_SPACE(fl))  // \S
    return true;
  if ((builtins & 32) && ((ch != '\n') && !IS_DIGIT(fl)))  // \D
    return true;
  //
  return false;
}

__device__ __forceinline__ reinst reprog_device::get_inst(int32_t id) const { return _insts[id]; }

__device__ __forceinline__ reclass_device reprog_device::get_class(int32_t id) const
{
  return _classes[id];
}

__device__ __forceinline__ bool reprog_device::is_empty() const
{
  return insts_counts() == 0 || get_inst(0).type == END;
}

/**
 * @brief Evaluate a specific string against regex pattern compiled to this instance.
 *
 * This is the main function for executing the regex against an individual string.
 *
 * @param dstr String used for matching.
 * @param jnk State data object for this string.
 * @param[in,out] begin Character position to start evaluation. On return, it is the position of the
 * match.
 * @param[in,out] end Character position to stop evaluation. On return, it is the end of the matched
 * substring.
 * @param group_id Index of the group to match in a multi-group regex pattern.
 * @return >0 if match found
 */
__device__ __forceinline__ int32_t reprog_device::regexec(string_view const dstr,
                                                          reljunk jnk,
                                                          cudf::size_type& begin,
                                                          cudf::size_type& end,
                                                          cudf::size_type const group_id) const
{
  int32_t match       = 0;
  auto pos            = begin;
  auto eos            = end;
  char_utf8 c         = 0;
  auto checkstart     = jnk.starttype != 0;
  auto last_character = false;

  string_view::const_iterator itr = string_view::const_iterator(dstr, pos);

  jnk.list1->reset();
  do {
    // fast check for first CHAR or BOL
    if (checkstart) {
      auto startchar = static_cast<char_utf8>(jnk.startchar);
      switch (jnk.starttype) {
        case BOL:
          if (pos == 0) break;
          if (jnk.startchar != '^') { return match; }
          --pos;
          startchar = static_cast<char_utf8>('\n');
        case CHAR: {
          auto const fidx = dstr.find(startchar, pos);
          if (fidx < 0) { return match; }
          pos = fidx + (jnk.starttype == BOL);
          break;
        }
      }
      itr += (pos - itr.position());  // faster to increment position
    }

    if (((eos < 0) || (pos < eos)) && match == 0) {
      auto ids = _startinst_ids;
      while (*ids >= 0)
        jnk.list1->activate(*ids++, (group_id == 0 ? pos : -1), -1);
    }

    last_character = itr.byte_offset() >= dstr.size_bytes();

    c = last_character ? 0 : *itr;

    // expand the non-character types like: LBRA, RBRA, BOL, EOL, BOW, NBOW, and OR
    bool expanded = false;
    do {
      jnk.list2->reset();
      expanded = false;

      for (int16_t i = 0; i < jnk.list1->get_size(); i++) {
        auto state          = jnk.list1->get_state(i);
        auto range          = state.range;
        auto const inst     = get_inst(state.inst_id);
        int32_t id_activate = -1;

        switch (inst.type) {
          case CHAR:
          case ANY:
          case ANYNL:
          case CCLASS:
          case NCCLASS:
          case END: id_activate = state.inst_id; break;
          case LBRA:
            if (inst.u1.subid == group_id) range.x = pos;
            id_activate = inst.u2.next_id;
            expanded    = true;
            break;
          case RBRA:
            if (inst.u1.subid == group_id) range.y = pos;
            id_activate = inst.u2.next_id;
            expanded    = true;
            break;
          case BOL:
            if ((pos == 0) || ((inst.u1.c == '^') && (dstr[pos - 1] == '\n'))) {
              id_activate = inst.u2.next_id;
              expanded    = true;
            }
            break;
          case EOL:
            if (last_character || (c == '\n' && inst.u1.c == '$')) {
              id_activate = inst.u2.next_id;
              expanded    = true;
            }
            break;
          case BOW:
          case NBOW: {
            auto const codept      = utf8_to_codepoint(c);
            auto const last_c      = pos > 0 ? dstr[pos - 1] : 0;
            auto const last_codept = utf8_to_codepoint(last_c);

            bool const cur_alphaNumeric =
              (codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[codept]);
            bool const last_alphaNumeric =
              (last_codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[last_codept]);
            if ((cur_alphaNumeric == last_alphaNumeric) != (inst.type == BOW)) {
              id_activate = inst.u2.next_id;
              expanded    = true;
            }
            break;
          }
          case OR:
            jnk.list2->activate(inst.u1.right_id, range.x, range.y);
            id_activate = inst.u2.left_id;
            expanded    = true;
            break;
        }
        if (id_activate >= 0) jnk.list2->activate(id_activate, range.x, range.y);
      }
      jnk.swaplist();

    } while (expanded);

    // execute instructions
    bool continue_execute = true;
    jnk.list2->reset();
    for (int16_t i = 0; continue_execute && i < jnk.list1->get_size(); i++) {
      auto const state    = jnk.list1->get_state(i);
      auto const range    = state.range;
      auto const inst     = get_inst(state.inst_id);
      int32_t id_activate = -1;

      switch (inst.type) {
        case CHAR:
          if (inst.u1.c == c) id_activate = inst.u2.next_id;
          break;
        case ANY:
          if (c != '\n') id_activate = inst.u2.next_id;
          break;
        case ANYNL: id_activate = inst.u2.next_id; break;
        case NCCLASS:
        case CCLASS: {
          auto const cls = get_class(inst.u1.cls_id);
          if (cls.is_match(static_cast<char32_t>(c), _codepoint_flags) == (inst.type == CCLASS)) {
            id_activate = inst.u2.next_id;
          }
          break;
        }
        case END:
          match = 1;
          begin = range.x;
          end   = group_id == 0 ? pos : range.y;
          // done with execute
          continue_execute = false;
          break;
      }
      if (continue_execute && (id_activate >= 0))
        jnk.list2->activate(id_activate, range.x, range.y);
    }

    ++pos;
    ++itr;
    jnk.swaplist();
    checkstart = jnk.list1->get_size() == 0;
  } while (!last_character && (!checkstart || !match));

  return match;
}

template <int stack_size>
__device__ __forceinline__ int32_t reprog_device::find(int32_t idx,
                                                       string_view const dstr,
                                                       cudf::size_type& begin,
                                                       cudf::size_type& end) const
{
  int32_t rtn = call_regexec<stack_size>(idx, dstr, begin, end);
  if (rtn <= 0) begin = end = -1;
  return rtn;
}

template <int stack_size>
__device__ __forceinline__ match_result reprog_device::extract(cudf::size_type idx,
                                                               string_view const dstr,
                                                               cudf::size_type begin,
                                                               cudf::size_type end,
                                                               cudf::size_type const group_id) const
{
  end = begin + 1;
  return call_regexec<stack_size>(idx, dstr, begin, end, group_id + 1) > 0
           ? match_result({begin, end})
           : thrust::nullopt;
}

template <int stack_size>
__device__ __forceinline__ int32_t reprog_device::call_regexec(int32_t idx,
                                                               string_view const dstr,
                                                               cudf::size_type& begin,
                                                               cudf::size_type& end,
                                                               cudf::size_type const group_id) const
{
  u_char data1[stack_size], data2[stack_size];

  relist list1(static_cast<int16_t>(_insts_count), data1);
  relist list2(static_cast<int16_t>(_insts_count), data2);

  reljunk jnk(&list1, &list2, get_inst(_startinst_id));
  return regexec(dstr, jnk, begin, end, group_id);
}

template <>
__device__ __forceinline__ int32_t
reprog_device::call_regexec<RX_STACK_ANY>(int32_t idx,
                                          string_view const dstr,
                                          cudf::size_type& begin,
                                          cudf::size_type& end,
                                          cudf::size_type const group_id) const
{
  auto const relists_size = relist::alloc_size(_insts_count);
  auto* listmem           = reinterpret_cast<u_char*>(_relists_mem);  // beginning of relist buffer;
  listmem += (idx * relists_size * 2);                                // two relist ptrs in reljunk:

  auto* list1 = new (listmem) relist(static_cast<int16_t>(_insts_count));
  auto* list2 = new (listmem + relists_size) relist(static_cast<int16_t>(_insts_count));

  reljunk jnk(list1, list2, get_inst(_startinst_id));
  return regexec(dstr, jnk, begin, end, group_id);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
