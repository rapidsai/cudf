/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/glushkov_regcomp.h"
// #include "strings/regex/regcomp.h"
#include "strings/regex/glushkov.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/char_tables.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstring>
#include <functional>
#include <numeric>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Pack a gkprog into a contiguous device buffer and
 *        return a device pointer to the embedded gkprog_device struct.
 *
 * Buffer layout:
 *   [gkprog_device struct]
 *   [_positions    : num_states × glushkov_position]
 *   [_shift_masks  : GLUSHKOV_MAX_SHIFTS_DEV × glushkov_state_t]
 *   [_shift_amounts: GLUSHKOV_MAX_SHIFTS_DEV × uint8_t + padding]
 *   [_reach_ascii  : GLUSHKOV_ASCII_TABLE_SIZE × glushkov_state_t]
 *   [_exception_successors : num_states × glushkov_state_t]
 *   [_classes : classes_count × reclass_device + variable-length literals]
 *
 * Returns {device_ptr_to_gkprog_device, raw_buffer_to_delete}
 * where the device ptr is the beginning of the buffer (cast as the struct).
 */
std::unique_ptr<gkprog_device, std::function<void(gkprog_device*)>> gkprog_device::create(
  gkprog const& h_gp, rmm::cuda_stream_view stream)
{
  uint32_t const num_states = h_gp.num_states;
  uint32_t const num_shifts = h_gp.shift_count;
  int32_t const classes_cnt = static_cast<int32_t>(h_gp.classes.size());

  // ---- Compute cumulative section offsets from buffer start ---------------
  // gkprog_device is allocated on the host with plain new; only the data
  // arrays live in the device buffer.
  std::size_t off = 0;

  off                       = cudf::util::round_up_unsafe(off, alignof(reinst));
  std::size_t const pos_off = off;
  off += num_states * sizeof(reinst);

  // _shift_masks: 8-byte aligned
  off                          = cudf::util::round_up_unsafe(off, alignof(glushkov_state_t));
  std::size_t const smasks_off = off;
  off += GLUSHKOV_MAX_SHIFTS * sizeof(glushkov_state_t);

  // _shift_amounts: 1-byte aligned (uint8_t array)
  std::size_t const samts_off = off;
  off += GLUSHKOV_MAX_SHIFTS * sizeof(uint8_t);

  // _reach_ascii: 8-byte aligned (GLUSHKOV_ASCII_TABLE_SIZE × glushkov_state_t = 1 KB)
  off                               = cudf::util::round_up_unsafe(off, alignof(glushkov_state_t));
  std::size_t const reach_ascii_off = off;
  off += GLUSHKOV_ASCII_TABLE_SIZE * sizeof(glushkov_state_t);

  // _exception_successors: 8-byte aligned
  off                       = cudf::util::round_up_unsafe(off, alignof(glushkov_state_t));
  std::size_t const exc_off = off;
  off += num_states * sizeof(glushkov_state_t);

  // _classes: 16-byte aligned (reclass_device is alignas(16))
  off                       = cudf::util::round_up_unsafe(off, alignof(reclass_device));
  std::size_t const cls_off = off;
  std::size_t cls_size      = static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device);
  for (auto const& cls : h_gp.classes) {
    cls_size += cls.literals.size() * sizeof(reclass_range);
  }
  off += cls_size;
  off = cudf::util::round_up_unsafe(off, sizeof(char32_t));

  std::size_t const total = off;

  // Allocate flat host + device buffers
  auto h_buf = cudf::detail::make_host_vector<u_char>(total, stream);
  std::memset(h_buf.data(), 0, total);
  auto* d_raw = new rmm::device_uvector<u_char>(total, stream);

  u_char* const h_base = h_buf.data();
  u_char* const d_base = d_raw->data();

  // ---- Place gkprog_device header --------------------------------
  // Allocated with plain new so it survives past h_buf's lifetime.
  auto* h_gp_dev           = new gkprog_device{};
  h_gp_dev->num_states     = num_states;
  h_gp_dev->shift_count    = num_shifts;
  h_gp_dev->first_set      = h_gp.first_set;
  h_gp_dev->accept_mask    = h_gp.accept_mask;
  h_gp_dev->exception_mask = h_gp.exception_mask;
  // h_gp_dev->nullable         = h_gp.nullable;
  // h_gp_dev->has_startchar    = h_gp.has_startchar;
  h_gp_dev->startchar        = h_gp.startchar;
  h_gp_dev->_codepoint_flags = get_character_flags_table(stream);
  h_gp_dev->_prog_size       = total;

  // ---- _positions ---------------------------------------------------------
  // h_gp_dev->_positions = reinterpret_cast<glushkov_position const*>(d_base + pos_off);
  // auto* pos_arr        = reinterpret_cast<glushkov_position*>(h_base + pos_off);
  h_gp_dev->_positions = reinterpret_cast<reinst const*>(d_base + pos_off);
  auto* pos_arr        = reinterpret_cast<reinst*>(h_base + pos_off);
  for (uint32_t i = 0; i < num_states; ++i) {  // memcpy could replace this now
    // auto t     = h_gp.pos_inst_type[i];
    // pos_arr[i] = t == CCLASS || t == NCCLASS
    //                ? reinst{.type = t, .u1 = {.cls_id = h_gp.pos_cls_idx[i]}}
    //                : reinst{.type = t, .u1 = {.c = h_gp.pos_ch[i]}};
    pos_arr[i] = h_gp.pos_insts[i];
  }

  // ---- _shift_masks --------------------------------------------------------
  h_gp_dev->_shift_masks = reinterpret_cast<glushkov_state_t const*>(d_base + smasks_off);
  std::memcpy(
    h_base + smasks_off, h_gp.shift_masks.data(), GLUSHKOV_MAX_SHIFTS * sizeof(glushkov_state_t));

  // ---- _shift_amounts ------------------------------------------------------
  h_gp_dev->_shift_amounts = reinterpret_cast<uint8_t const*>(d_base + samts_off);
  std::memcpy(h_base + samts_off, h_gp.shift_amounts.data(), GLUSHKOV_MAX_SHIFTS * sizeof(uint8_t));

  // ---- _reach_ascii -------------------------------------------------------
  h_gp_dev->_reach_ascii = reinterpret_cast<glushkov_state_t const*>(d_base + reach_ascii_off);
  std::memcpy(h_base + reach_ascii_off,
              h_gp.reach_ascii.data(),
              GLUSHKOV_ASCII_TABLE_SIZE * sizeof(glushkov_state_t));

  // ---- _exception_successors ---------------------------------------------------
  h_gp_dev->_exception_successors = reinterpret_cast<glushkov_state_t const*>(d_base + exc_off);
  std::memcpy(
    h_base + exc_off, h_gp.exception_successors.data(), num_states * sizeof(glushkov_state_t));

  // ---- _classes (variable-length, same layout as Thompson reprog_device) --
  h_gp_dev->_classes = reinterpret_cast<reclass_device const*>(d_base + cls_off);
  auto* h_cls        = reinterpret_cast<reclass_device*>(h_base + cls_off);
  u_char* h_lit = h_base + cls_off + static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device);
  u_char* d_lit = d_base + cls_off + static_cast<std::size_t>(classes_cnt) * sizeof(reclass_device);
  for (int32_t ci = 0; ci < classes_cnt; ++ci) {
    auto const& src             = h_gp.classes[ci];
    *h_cls++                    = reclass_device{src.builtins,
                              static_cast<int32_t>(src.literals.size()),
                              reinterpret_cast<reclass_range const*>(d_lit)};
    std::size_t const lit_bytes = src.literals.size() * sizeof(reclass_range);
    std::memcpy(h_lit, src.literals.data(), lit_bytes);
    h_lit += lit_bytes;
    d_lit += lit_bytes;
  }

  // copy flat buffer to device
  cudf::detail::cuda_memcpy_async<u_char>(*d_raw, h_buf, stream);

  auto deleter = [d_raw](gkprog_device* t) {
    t->destroy();
    delete d_raw;
  };

  return std::unique_ptr<gkprog_device, std::function<void(gkprog_device*)>>(h_gp_dev, deleter);
}

void gkprog_device::destroy() { delete this; }

int32_t gkprog_device::compute_shared_memory_size() const
{
  return static_cast<int32_t>(sizeof(glushkov_shmem_cache));
}

std::pair<std::size_t, int32_t> gkprog_device::compute_strided_working_memory(int32_t rows,
                                                                              int32_t,
                                                                              std::size_t) const
{
  return std::make_pair(0, rows);
}

void gkprog_device::set_working_memory(void*, int32_t thread_count, int32_t)
{
  _thread_count = thread_count;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
