/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/regex/glushkov_regcomp.h"
#include "strings/regex/regcomp.h"
#include "strings/regex/regex.cuh"

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

// ===========================================================================
// Thompson NFA device program creation
// ===========================================================================

// Copy reprog primitive values
reprog_device::reprog_device(reprog const& prog)
  : _startinst_id{prog.get_start_inst()},
    _num_capturing_groups{prog.groups_count()},
    _insts_count{prog.insts_count()},
    _starts_count{prog.starts_count()},
    _classes_count{prog.classes_count()},
    _max_insts{prog.insts_count()}
{
}

// ---------------------------------------------------------------------------
// Glushkov device program builder
// ---------------------------------------------------------------------------

/**
 * @brief Pack a glushkov_host_program into a contiguous device buffer and
 *        return a device pointer to the embedded glushkov_program_device struct.
 *
 * Buffer layout:
 *   [glushkov_program_device struct]
 *   [_positions    : num_states × glushkov_position]
 *   [_shift_masks  : GLUSHKOV_MAX_SHIFTS_DEV × g_state_t]
 *   [_shift_amounts: GLUSHKOV_MAX_SHIFTS_DEV × uint8_t + padding]
 *   [_reach_ascii  : GLUSHKOV_ASCII_TABLE_SIZE × g_state_t]
 *   [_exception_succs : num_states × g_state_t]
 *   [_classes : classes_count × reclass_device + variable-length literals]
 *
 * Returns {device_ptr_to_glushkov_program_device, raw_buffer_to_delete}
 * where the device ptr is the beginning of the buffer (cast as the struct).
 */
static std::pair<glushkov_program_device const*, rmm::device_uvector<u_char>*>
create_glushkov_device(glushkov_host_program const& h_gp,
                       uint8_t const* d_codepoint_flags,
                       rmm::cuda_stream_view stream)
{
  uint32_t const num_states = h_gp.num_states;
  uint32_t const num_shifts = h_gp.shift_count;
  int32_t const classes_cnt = static_cast<int32_t>(h_gp.classes.size());

  // ---- Compute cumulative section offsets from buffer start ---------------
  // This ensures each section's ABSOLUTE address (buf_base + offset) satisfies
  // its alignment requirement regardless of sizeof(glushkov_program_device).
  std::size_t off = 0;

  // Header
  std::size_t const hdr_off = off;
  off += sizeof(glushkov_program_device);

  // _positions: 4-byte aligned (int32_t / char32_t fields)
  off                       = cudf::util::round_up_unsafe(off, alignof(glushkov_position));
  std::size_t const pos_off = off;
  off += num_states * sizeof(glushkov_position);

  // _shift_masks: 8-byte aligned
  off                          = cudf::util::round_up_unsafe(off, alignof(g_state_t));
  std::size_t const smasks_off = off;
  off += GLUSHKOV_MAX_SHIFTS_DEV * sizeof(g_state_t);

  // _shift_amounts: 1-byte aligned (uint8_t array)
  std::size_t const samts_off = off;
  off += GLUSHKOV_MAX_SHIFTS_DEV * sizeof(uint8_t);

  // _reach_ascii: 8-byte aligned (GLUSHKOV_ASCII_TABLE_SIZE × g_state_t = 1 KB)
  off                               = cudf::util::round_up_unsafe(off, alignof(g_state_t));
  std::size_t const reach_ascii_off = off;
  off += GLUSHKOV_ASCII_TABLE_SIZE * sizeof(g_state_t);

  // _exception_succs: 8-byte aligned
  off                       = cudf::util::round_up_unsafe(off, alignof(g_state_t));
  std::size_t const exc_off = off;
  off += num_states * sizeof(g_state_t);

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

  // ---- Place glushkov_program_device header --------------------------------
  auto* h_gp_dev             = new (h_base + hdr_off) glushkov_program_device{};
  h_gp_dev->num_states       = num_states;
  h_gp_dev->shift_count      = num_shifts;
  h_gp_dev->first_set        = h_gp.first_set;
  h_gp_dev->accept_mask      = h_gp.accept_mask;
  h_gp_dev->exception_mask   = h_gp.exception_mask;
  h_gp_dev->nullable         = h_gp.nullable;
  h_gp_dev->has_startchar    = h_gp.has_startchar;
  h_gp_dev->startchar        = h_gp.startchar;
  h_gp_dev->_codepoint_flags = d_codepoint_flags;
  h_gp_dev->_prog_size       = total;

  // ---- _positions ---------------------------------------------------------
  h_gp_dev->_positions = reinterpret_cast<glushkov_position const*>(d_base + pos_off);
  auto* pos_arr        = reinterpret_cast<glushkov_position*>(h_base + pos_off);
  for (uint32_t i = 0; i < num_states; ++i) {
    pos_arr[i] = {h_gp.pos_inst_type[i], h_gp.pos_ch[i], h_gp.pos_cls_idx[i]};
  }

  // ---- _shift_masks --------------------------------------------------------
  h_gp_dev->_shift_masks = reinterpret_cast<g_state_t const*>(d_base + smasks_off);
  std::memcpy(
    h_base + smasks_off, h_gp.shift_masks.data(), GLUSHKOV_MAX_SHIFTS_DEV * sizeof(g_state_t));

  // ---- _shift_amounts ------------------------------------------------------
  h_gp_dev->_shift_amounts = reinterpret_cast<uint8_t const*>(d_base + samts_off);
  std::memcpy(
    h_base + samts_off, h_gp.shift_amounts.data(), GLUSHKOV_MAX_SHIFTS_DEV * sizeof(uint8_t));

  // ---- _reach_ascii -------------------------------------------------------
  h_gp_dev->_reach_ascii = reinterpret_cast<g_state_t const*>(d_base + reach_ascii_off);
  std::memcpy(h_base + reach_ascii_off,
              h_gp.reach_ascii.data(),
              GLUSHKOV_ASCII_TABLE_SIZE * sizeof(g_state_t));

  // ---- _exception_succs ---------------------------------------------------
  h_gp_dev->_exception_succs = reinterpret_cast<g_state_t const*>(d_base + exc_off);
  std::memcpy(h_base + exc_off, h_gp.exception_succs.data(), num_states * sizeof(g_state_t));

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

  // Copy flat buffer to device
  cudf::detail::cuda_memcpy_async<u_char>(*d_raw, h_buf, stream);

  // The glushkov_program_device struct lives at the start of the device buffer
  auto* d_gp_ptr = reinterpret_cast<glushkov_program_device const*>(d_base);
  return {d_gp_ptr, d_raw};
}

// ---------------------------------------------------------------------------
// reprog_device::create  (extended to optionally build Glushkov program)
// ---------------------------------------------------------------------------

std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> reprog_device::create(
  reprog const& h_prog, rmm::cuda_stream_view stream)
{
  // compute size to hold all the member data
  return reprog_device::create(h_prog, nullptr, stream);
}

std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> reprog_device::create(
  reprog const& h_prog, glushkov_host_program const* h_glushkov, rmm::cuda_stream_view stream)
{
  // ---- Thompson NFA: existing layout ----
  auto const insts_count   = h_prog.insts_count();
  auto const classes_count = h_prog.classes_count();
  auto const starts_count  = h_prog.starts_count();

  // compute size of each section
  auto insts_size    = insts_count * sizeof(_insts[0]);
  auto startids_size = starts_count * sizeof(_startinst_ids[0]);
  auto classes_size =
    std::transform_reduce(h_prog.classes_data(),
                          h_prog.classes_data() + h_prog.classes_count(),
                          classes_count * sizeof(_classes[0]),
                          std::plus<std::size_t>{},
                          [](auto& cls) { return cls.literals.size() * sizeof(reclass_range); });
  // make sure each section is aligned for the subsequent section's data type
  auto const memsize = cudf::util::round_up_safe(insts_size, sizeof(_startinst_ids[0])) +
                       cudf::util::round_up_safe(startids_size, sizeof(_classes[0])) +
                       cudf::util::round_up_safe(classes_size, sizeof(char32_t));

  // allocate memory to store all the prog data in a flat contiguous buffer
  auto h_buffer =
    cudf::detail::make_host_vector<u_char>(memsize, stream);  // copy everything into here;
  auto h_ptr    = h_buffer.data();                            // this is our running host ptr;
  auto d_buffer = new rmm::device_uvector<u_char>(memsize, stream);  // output device memory;
  auto d_ptr    = d_buffer->data();                                  // running device pointer

  // create our device object; this is managed separately and returned to the caller
  auto* d_prog = new reprog_device(h_prog);

  d_prog->_codepoint_flags = get_character_flags_table(stream);

  // copy the instructions array first (fixed-sized structs)
  memcpy(h_ptr, h_prog.insts_data(), insts_size);
  d_prog->_insts = reinterpret_cast<reinst*>(d_ptr);

  // point to the end for the next section
  insts_size = cudf::util::round_up_safe(insts_size, sizeof(_startinst_ids[0]));
  h_ptr += insts_size;
  d_ptr += insts_size;
  // copy the startinst_ids next
  memcpy(h_ptr, h_prog.starts_data(), startids_size);
  d_prog->_startinst_ids = reinterpret_cast<int32_t*>(d_ptr);

  // next section; align the size for next data type
  startids_size = cudf::util::round_up_safe(startids_size, sizeof(_classes[0]));
  h_ptr += startids_size;
  d_ptr += startids_size;
  // copy classes into flat memory: [class1,class2,...][char32 arrays]
  auto classes     = reinterpret_cast<reclass_device*>(h_ptr);
  d_prog->_classes = reinterpret_cast<reclass_device*>(d_ptr);
  // get pointer to the end to handle variable length data
  auto h_end = h_ptr + (classes_count * sizeof(reclass_device));
  auto d_end = d_ptr + (classes_count * sizeof(reclass_device));
  // place each class and append the variable length data
  for (int32_t idx = 0; idx < classes_count; ++idx) {
    auto const& h_class = h_prog.class_at(idx);
    reclass_device const d_class{h_class.builtins,
                                 static_cast<int32_t>(h_class.literals.size()),
                                 reinterpret_cast<reclass_range*>(d_end)};
    *classes++ = d_class;
    memcpy(h_end, h_class.literals.data(), h_class.literals.size() * sizeof(reclass_range));
    h_end += h_class.literals.size() * sizeof(reclass_range);
    d_end += h_class.literals.size() * sizeof(reclass_range);
  }

  // initialize the rest of the elements
  d_prog->_max_insts = insts_count;
  d_prog->_prog_size = memsize + sizeof(reprog_device);

  // copy flat prog to device memory
  cudf::detail::cuda_memcpy_async<u_char>(*d_buffer, h_buffer, stream);

  // ---- Optional Glushkov program ------------------------------------------
  rmm::device_uvector<u_char>* d_glushkov_buffer = nullptr;
  if (h_glushkov) {
    auto [d_gp_ptr, d_gb] = create_glushkov_device(*h_glushkov, d_prog->_codepoint_flags, stream);
    d_prog->_glushkov     = d_gp_ptr;
    d_glushkov_buffer     = d_gb;
  }

  // create a deleter to free both device buffers
  auto deleter = [d_buffer, d_glushkov_buffer](reprog_device* t) {
    t->destroy();
    delete d_buffer;
    delete d_glushkov_buffer;  // safe to delete nullptr
  };

  return std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>(d_prog, deleter);
}

void reprog_device::destroy() { delete this; }

// ---------------------------------------------------------------------------
// Working-memory helpers (Glushkov uses zero working memory)
// ---------------------------------------------------------------------------

std::size_t reprog_device::working_memory_size(int32_t num_threads) const
{
  // Glushkov uses register-only state (uint64_t bitmask) — zero working memory.
  // Extract APIs create their reprog_device without Glushkov (use_glushkov=false),
  // so they still get Thompson working memory allocated here.
  if (_glushkov) return 0;
  return compute_working_memory_size(num_threads, insts_counts());
}

std::pair<std::size_t, int32_t> reprog_device::compute_strided_working_memory(
  int32_t rows, int32_t min_rows, std::size_t requested_max_size) const
{
  auto thread_count = rows;
  auto buffer_size  = working_memory_size(thread_count);
  while ((buffer_size > requested_max_size) && (thread_count > min_rows)) {
    thread_count = thread_count / 2;
    buffer_size  = working_memory_size(thread_count);
  }
  // clamp to min_rows but only if rows is greater than min_rows
  if (rows > min_rows && thread_count < min_rows) {
    thread_count = min_rows;
    buffer_size  = working_memory_size(thread_count);
  }
  return std::make_pair(buffer_size, thread_count);
}

void reprog_device::set_working_memory(void* buffer, int32_t thread_count, int32_t max_insts)
{
  _buffer       = buffer;
  _thread_count = thread_count;
  _max_insts    = _max_insts > 0 ? _max_insts : _insts_count;
}

int32_t reprog_device::compute_shared_memory_size_thompson() const
{ return _prog_size < MAX_SHARED_MEM ? static_cast<int32_t>(_prog_size) : 0; }

int32_t reprog_device::compute_shared_memory_size() const
{
  auto size = compute_shared_memory_size_thompson();
  if (_glushkov) {
    size = cudf::util::round_up_unsafe(size, 8);
    size += static_cast<int32_t>(sizeof(glushkov_shmem_cache));
  }
  return size;
}

std::size_t compute_working_memory_size(int32_t num_threads, int32_t insts_count)
{ return relist::alloc_size(insts_count, num_threads) * 2; }

}  // namespace detail
}  // namespace strings
}  // namespace cudf
