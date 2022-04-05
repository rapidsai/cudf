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

#include <strings/regex/regcomp.h>
#include <strings/regex/regex.cuh>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {

// Copy reprog primitive values
reprog_device::reprog_device(reprog& prog)
  : _startinst_id{prog.get_start_inst()},
    _num_capturing_groups{prog.groups_count()},
    _insts_count{prog.insts_count()},
    _starts_count{prog.starts_count()},
    _classes_count{prog.classes_count()}
{
}

std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> reprog_device::create(
  std::string const& pattern,
  uint8_t const* codepoint_flags,
  size_type strings_count,
  rmm::cuda_stream_view stream)
{
  return reprog_device::create(
    pattern, regex_flags::MULTILINE, codepoint_flags, strings_count, stream);
}

// Create instance of the reprog that can be passed into a device kernel
std::unique_ptr<reprog_device, std::function<void(reprog_device*)>> reprog_device::create(
  std::string const& pattern,
  regex_flags const flags,
  uint8_t const* codepoint_flags,
  size_type strings_count,
  rmm::cuda_stream_view stream)
{
  // compile pattern into host object
  reprog h_prog = reprog::create_from(pattern, flags);

  // compute size to hold all the member data
  auto const insts_count   = h_prog.insts_count();
  auto const classes_count = h_prog.classes_count();
  auto const starts_count  = h_prog.starts_count();

  // compute size of each section; make sure each is aligned appropriately
  auto insts_size    = insts_count * sizeof(_insts[0]);
  auto startids_size = starts_count * sizeof(_startinst_ids[0]);
  auto classes_size  = classes_count * sizeof(_classes[0]);
  for (auto idx = 0; idx < classes_count; ++idx)
    classes_size += static_cast<int32_t>((h_prog.class_at(idx).literals.size()) * sizeof(char32_t));
  // make sure each section is aligned for the subsequent section's data type
  auto const memsize = cudf::util::round_up_safe(insts_size, sizeof(_startinst_ids[0])) +
                       cudf::util::round_up_safe(startids_size, sizeof(_classes[0])) +
                       cudf::util::round_up_safe(classes_size, sizeof(char32_t));

  // allocate memory to store all the prog data in a flat contiguous buffer
  std::vector<u_char> h_buffer(memsize);                        // copy everything into here;
  auto h_ptr    = h_buffer.data();                              // this is our running host ptr;
  auto d_buffer = new rmm::device_buffer(memsize, stream);      // output device memory;
  auto d_ptr    = reinterpret_cast<u_char*>(d_buffer->data());  // running device pointer

  // put everything into a flat host buffer first
  reprog_device* d_prog = new reprog_device(h_prog);

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
    reclass& h_class = h_prog.class_at(idx);
    reclass_device d_class{h_class.builtins,
                           static_cast<int32_t>(h_class.literals.size() / 2),
                           reinterpret_cast<char32_t*>(d_end)};
    *classes++ = d_class;
    memcpy(h_end, h_class.literals.c_str(), h_class.literals.size() * sizeof(char32_t));
    h_end += h_class.literals.size() * sizeof(char32_t);
    d_end += h_class.literals.size() * sizeof(char32_t);
  }

  // initialize the rest of the elements
  d_prog->_codepoint_flags = codepoint_flags;

  // allocate execute memory if needed
  rmm::device_buffer* d_relists{};
  if (insts_count > RX_LARGE_INSTS) {
    // two relist state structures are needed for execute per string
    auto const rlm_size  = relist::alloc_size(insts_count) * 2 * strings_count;
    d_relists            = new rmm::device_buffer(rlm_size, stream);
    d_prog->_relists_mem = d_relists->data();
  }

  // copy flat prog to device memory
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_buffer->data(), h_buffer.data(), memsize, cudaMemcpyHostToDevice, stream.value()));

  // build deleter to cleanup device memory
  auto deleter = [d_buffer, d_relists](reprog_device* t) {
    t->destroy();
    delete d_buffer;
    delete d_relists;
  };
  return std::unique_ptr<reprog_device, std::function<void(reprog_device*)>>(d_prog, deleter);
}

void reprog_device::destroy() { delete this; }

}  // namespace detail
}  // namespace strings
}  // namespace cudf
