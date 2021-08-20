/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>

#include <memory.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>

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
  int16_t size{};
  int16_t listsize{};
  int32_t reserved;
  int2* ranges{};       // pair per instruction
  int16_t* inst_ids{};  // one per instruction
  u_char* mask{};       // bit per instruction

  __host__ __device__ inline static int32_t data_size_for(int32_t insts)
  {
    return ((sizeof(ranges[0]) + sizeof(inst_ids[0])) * insts) + ((insts + 7) / 8);
  }

  __host__ __device__ inline static int32_t alloc_size(int32_t insts)
  {
    int32_t size = sizeof(relist);
    size += data_size_for(insts);
    size = ((size + 7) / 8) * 8;  // align it too
    return size;
  }

  __host__ __device__ inline relist() {}

  __host__ __device__ inline relist(int16_t insts, u_char* data = nullptr) : listsize(insts)
  {
    auto ptr = data == nullptr ? reinterpret_cast<u_char*>(this) + sizeof(relist) : data;
    ranges   = reinterpret_cast<int2*>(ptr);
    ptr += listsize * sizeof(ranges[0]);
    inst_ids = reinterpret_cast<int16_t*>(ptr);
    ptr += listsize * sizeof(inst_ids[0]);
    mask = ptr;
    reset();
  }

  __host__ __device__ inline void reset()
  {
    memset(mask, 0, (listsize + 7) / 8);
    size = 0;
  }

  __device__ inline bool activate(int32_t i, int32_t begin, int32_t end)
  {
    if (readMask(i)) return false;
    writeMask(true, i);
    inst_ids[size] = static_cast<int16_t>(i);
    ranges[size]   = int2{begin, end};
    ++size;
    return true;
  }

  __device__ inline void writeMask(bool v, int32_t pos)
  {
    u_char uc = 1 << (pos & 7);
    if (v)
      mask[pos >> 3] |= uc;
    else
      mask[pos >> 3] &= ~uc;
  }

  __device__ inline bool readMask(int32_t pos)
  {
    u_char uc = mask[pos >> 3];
    return static_cast<bool>((uc >> (pos & 7)) & 1);
  }
};

/**
 * @brief This manages the two relist instances required by the regexec function.
 */
struct reljunk {
  relist* list1;
  relist* list2;
  int32_t starttype{};
  char32_t startchar{};

  __host__ __device__ reljunk(relist* list1, relist* list2, int32_t stype, char32_t schar)
    : list1(list1), list2(list2)
  {
    if (starttype == CHAR || starttype == BOL) {
      starttype = stype;
      startchar = schar;
    }
  }
};

__device__ inline void swaplist(relist*& l1, relist*& l2)
{
  relist* tmp = l1;
  l1          = l2;
  l2          = tmp;
}

/**
 * @brief Utility to check a specific character against this class instance.
 *
 * @param ch A 4-byte UTF-8 character.
 * @param codepoint_flags Used for mapping a character to type for builtin classes.
 * @return true if the character matches
 */
__device__ inline bool reclass_device::is_match(char32_t ch, const uint8_t* codepoint_flags)
{
  if (thrust::any_of(thrust::seq,
                     thrust::make_counting_iterator<int>(0),
                     thrust::make_counting_iterator<int>(count),
                     [ch, this] __device__(int i) {
                       return ((ch >= literals[i * 2]) && (ch <= literals[(i * 2) + 1]));
                     }))
    return true;
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

__device__ inline reinst* reprog_device::get_inst(int32_t idx) const
{
  assert((idx >= 0) && (idx < _insts_count));
  return _insts + idx;
}

__device__ inline reclass_device reprog_device::get_class(int32_t idx) const
{
  assert((idx >= 0) && (idx < _classes_count));
  return _classes[idx];
}

__device__ inline int32_t* reprog_device::startinst_ids() const { return _startinst_ids; }

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
__device__ inline int32_t reprog_device::regexec(
  string_view const& dstr, reljunk& jnk, int32_t& begin, int32_t& end, int32_t group_id)
{
  int32_t match                   = 0;
  auto checkstart                 = jnk.starttype;
  auto txtlen                     = dstr.length();
  auto pos                        = begin;
  auto eos                        = end;
  char32_t c                      = 0;
  string_view::const_iterator itr = string_view::const_iterator(dstr, pos);

  jnk.list1->reset();
  do {
    /* fast check for first char */
    if (checkstart) {
      switch (jnk.starttype) {
        case CHAR: {
          auto fidx = dstr.find(static_cast<char_utf8>(jnk.startchar), pos);
          if (fidx < 0) return match;
          pos = fidx;
          break;
        }
        case BOL: {
          if (pos == 0) break;
          if (jnk.startchar != '^') return match;
          --pos;
          int fidx = dstr.find(static_cast<char_utf8>('\n'), pos);
          if (fidx < 0) return match;  // update begin/end values?
          pos = fidx + 1;
          break;
        }
      }
      itr += (pos - itr.position());  // faster to increment position
    }

    if (((eos < 0) || (pos < eos)) && match == 0) {
      int32_t i = 0;
      auto ids  = startinst_ids();
      while (ids[i] >= 0)
        jnk.list1->activate(ids[i++], (group_id == 0 ? pos : -1), -1);
    }

    c = static_cast<char32_t>(pos >= txtlen ? 0 : *itr);

    // expand LBRA, RBRA, BOL, EOL, BOW, NBOW, and OR
    bool expanded = false;
    do {
      jnk.list2->reset();
      expanded = false;

      for (int16_t i = 0; i < jnk.list1->size; i++) {
        int32_t inst_id     = static_cast<int32_t>(jnk.list1->inst_ids[i]);
        int2& range         = jnk.list1->ranges[i];
        const reinst* inst  = get_inst(inst_id);
        int32_t id_activate = -1;

        switch (inst->type) {
          case CHAR:
          case ANY:
          case ANYNL:
          case CCLASS:
          case NCCLASS:
          case END: id_activate = inst_id; break;
          case LBRA:
            if (inst->u1.subid == group_id) range.x = pos;
            id_activate = inst->u2.next_id;
            expanded    = true;
            break;
          case RBRA:
            if (inst->u1.subid == group_id) range.y = pos;
            id_activate = inst->u2.next_id;
            expanded    = true;
            break;
          case BOL:
            if ((pos == 0) ||
                ((inst->u1.c == '^') && (dstr[pos - 1] == static_cast<char_utf8>('\n')))) {
              id_activate = inst->u2.next_id;
              expanded    = true;
            }
            break;
          case EOL:
            if ((c == 0) || (inst->u1.c == '$' && c == '\n')) {
              id_activate = inst->u2.next_id;
              expanded    = true;
            }
            break;
          case BOW: {
            auto codept           = utf8_to_codepoint(c);
            char32_t last_c       = static_cast<char32_t>(pos ? dstr[pos - 1] : 0);
            auto last_codept      = utf8_to_codepoint(last_c);
            bool cur_alphaNumeric = (codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[codept]);
            bool last_alphaNumeric =
              (last_codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[last_codept]);
            if (cur_alphaNumeric != last_alphaNumeric) {
              id_activate = inst->u2.next_id;
              expanded    = true;
            }
            break;
          }
          case NBOW: {
            auto codept           = utf8_to_codepoint(c);
            char32_t last_c       = static_cast<char32_t>(pos ? dstr[pos - 1] : 0);
            auto last_codept      = utf8_to_codepoint(last_c);
            bool cur_alphaNumeric = (codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[codept]);
            bool last_alphaNumeric =
              (last_codept < 0x010000) && IS_ALPHANUM(_codepoint_flags[last_codept]);
            if (cur_alphaNumeric == last_alphaNumeric) {
              id_activate = inst->u2.next_id;
              expanded    = true;
            }
            break;
          }
          case OR:
            jnk.list2->activate(inst->u1.right_id, range.x, range.y);
            id_activate = inst->u2.left_id;
            expanded    = true;
            break;
        }
        if (id_activate >= 0) jnk.list2->activate(id_activate, range.x, range.y);
      }
      swaplist(jnk.list1, jnk.list2);

    } while (expanded);

    // execute
    bool continue_execute = true;
    jnk.list2->reset();
    for (int16_t i = 0; continue_execute && i < jnk.list1->size; i++) {
      int32_t inst_id     = static_cast<int32_t>(jnk.list1->inst_ids[i]);
      int2& range         = jnk.list1->ranges[i];
      const reinst* inst  = get_inst(inst_id);
      int32_t id_activate = -1;

      switch (inst->type) {
        case CHAR:
          if (inst->u1.c == c) id_activate = inst->u2.next_id;
          break;
        case ANY:
          if (c != '\n') id_activate = inst->u2.next_id;
          break;
        case ANYNL: id_activate = inst->u2.next_id; break;
        case CCLASS: {
          reclass_device cls = get_class(inst->u1.cls_id);
          if (cls.is_match(c, _codepoint_flags)) id_activate = inst->u2.next_id;
          break;
        }
        case NCCLASS: {
          reclass_device cls = get_class(inst->u1.cls_id);
          if (!cls.is_match(c, _codepoint_flags)) id_activate = inst->u2.next_id;
          break;
        }
        case END:
          match = 1;
          begin = range.x;
          end   = group_id == 0 ? pos : range.y;

          continue_execute = false;
          break;
      }
      if (continue_execute && (id_activate >= 0))
        jnk.list2->activate(id_activate, range.x, range.y);
    }

    ++pos;
    ++itr;
    swaplist(jnk.list1, jnk.list2);
    checkstart = jnk.list1->size > 0 ? 0 : 1;
  } while (c && (jnk.list1->size > 0 || match == 0));

  return match;
}

template <int stack_size>
__device__ inline int32_t reprog_device::find(int32_t idx,
                                              string_view const& dstr,
                                              int32_t& begin,
                                              int32_t& end)
{
  int32_t rtn = call_regexec<stack_size>(idx, dstr, begin, end);
  if (rtn <= 0) begin = end = -1;
  return rtn;
}

template <int stack_size>
__device__ inline match_result reprog_device::extract(cudf::size_type idx,
                                                      string_view const& dstr,
                                                      cudf::size_type begin,
                                                      cudf::size_type end,
                                                      cudf::size_type group_id)
{
  end = begin + 1;
  return call_regexec<stack_size>(idx, dstr, begin, end, group_id + 1) > 0
           ? match_result({begin, end})
           : thrust::nullopt;
}

template <int stack_size>
__device__ inline int32_t reprog_device::call_regexec(
  int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t group_id)
{
  u_char data1[stack_size], data2[stack_size];

  auto const stype = get_inst(_startinst_id)->type;
  auto const schar = get_inst(_startinst_id)->u1.c;

  relist list1(static_cast<int16_t>(_insts_count), data1);
  relist list2(static_cast<int16_t>(_insts_count), data2);

  reljunk jnk(&list1, &list2, stype, schar);
  return regexec(dstr, jnk, begin, end, group_id);
}

template <>
__device__ inline int32_t reprog_device::call_regexec<RX_STACK_ANY>(
  int32_t idx, string_view const& dstr, int32_t& begin, int32_t& end, int32_t group_id)
{
  auto const stype = get_inst(_startinst_id)->type;
  auto const schar = get_inst(_startinst_id)->u1.c;

  auto const relists_size = relist::alloc_size(_insts_count);
  u_char* listmem         = reinterpret_cast<u_char*>(_relists_mem);  // beginning of relist buffer;
  listmem += (idx * relists_size * 2);                                // two relist ptrs in reljunk:

  relist* list1 = new (listmem) relist(static_cast<int16_t>(_insts_count));
  relist* list2 = new (listmem + relists_size) relist(static_cast<int16_t>(_insts_count));

  reljunk jnk(list1, list2, stype, schar);
  return regexec(dstr, jnk, begin, end, group_id);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
