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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/edit_distance.hpp>

#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

using strings_iterator = cudf::column_device_view::const_iterator<cudf::string_view>;

/**
 * @brief Compute the Levenshtein distance for each string.
 *
 * Documentation here: https://www.cuelogic.com/blog/the-levenshtein-algorithm
 * And here: https://en.wikipedia.org/wiki/Levenshtein_distances
 */
struct edit_distance_levenshtein_algorithm {
  cudf::column_device_view d_strings;  // computing these
  cudf::column_device_view d_targets;  // with these;
  int16_t* d_buffer;                   // compute buffer for each string
  int32_t const* d_offsets;            // locate sub-buffer for each string

  __device__ int32_t operator()(cudf::size_type idx)
  {
    auto d_str =
      d_strings.is_null(idx) ? cudf::string_view{} : d_strings.element<cudf::string_view>(idx);
    auto d_tgt = [&] __device__ {
      if (d_targets.is_null(idx)) return cudf::string_view{};
      return d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                   : d_targets.element<cudf::string_view>(idx);
    }();
    return compute_distance(d_str, d_tgt, d_buffer + d_offsets[idx]);
  }

  __device__ int32_t compute_distance(cudf::string_view const& d_str,
                                      cudf::string_view const& d_tgt,
                                      int16_t* buffer)
  {
    if (d_str.empty()) return d_tgt.length();
    if (d_tgt.empty()) return d_str.length();

    auto itr_A = d_str.begin();
    auto itr_B = d_tgt.begin();
    auto len_A = d_str.length();
    auto len_B = d_tgt.length();
    if (len_A > len_B) {
      len_B = len_A;
      len_A = d_tgt.length();
      itr_A = d_tgt.begin();
      itr_B = d_str.begin();
    }
    auto line2 = buffer;
    auto line1 = line2 + len_A;
    auto line0 = line1 + len_A;
    int range  = len_A + len_B - 1;
    for (int i = 0; i < range; i++) {
      auto tmp = line2;
      line2    = line1;
      line1    = line0;
      line0    = tmp;

      for (int x = (i < len_B ? 0 : i - len_B + 1); (x < len_A) && (x < i + 1); x++) {
        int y     = i - x;
        int16_t u = y > 0 ? line1[x] : x + 1;
        int16_t v = x > 0 ? line1[x - 1] : y + 1;
        int16_t w;
        if ((x > 0) && (y > 0))
          w = line2[x - 1];
        else if (x > y)
          w = x;
        else
          w = y;
        u++;
        v++;
        itr_A += (x - itr_A.position());
        itr_B += (y - itr_B.position());
        auto c1 = *itr_A;
        auto c2 = *itr_B;
        if (c1 != c2) w++;
        int16_t value = u;
        if (v < value) value = v;
        if (w < value) value = w;
        line0[x] = value;
      }
    }
    return static_cast<int32_t>(line0[len_A - 1]);
  }
};

}  // namespace

/**
 * @copydoc nvtext::edit_distance
 */
std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& strings,
                                            cudf::strings_column_view const& targets,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  cudf::size_type strings_count = strings.size();
  if (strings_count == 0) return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});
  if (targets.size() > 1)
    CUDF_EXPECTS(strings.size() == targets.size(), "targets.size() must equal strings.size()");

  // create device column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  auto targets_column = cudf::column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;

  // calculate the size of the compute-buffer
  // we can use the output column buffer to hold these values temporarily
  auto results   = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                               strings_count,
                                               rmm::device_buffer{0, stream, mr},
                                               0,
                                               stream,
                                               mr);
  auto d_results = results->mutable_view().data<int32_t>();
  auto execpol   = rmm::exec_policy(0);
  thrust::transform(
    execpol->on(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator<cudf::size_type>(strings_count),
    d_results,
    [d_strings, d_targets] __device__(auto idx) {
      if (d_strings.is_null(idx) || d_targets.is_null(idx)) return int32_t{0};
      auto d_str = d_strings.element<cudf::string_view>(idx);
      auto d_tgt = d_targets.size() == 1 ? d_targets.element<cudf::string_view>(0)
                                         : d_targets.element<cudf::string_view>(idx);
      // just need 3 int16's for each character of the shorter string
      return static_cast<int32_t>(std::min(d_str.length(), d_tgt.length()) * (3 * sizeof(int16_t)));
    });

  // get the total size
  size_t compute_size =
    thrust::reduce(execpol->on(stream), d_results, d_results + strings_count, size_t{0});

  // convert sizes to offsets in-place
  thrust::exclusive_scan(execpol->on(stream), d_results, d_results + strings_count, d_results);
  rmm::device_buffer compute_buffer(compute_size, stream);
  auto d_buffer = reinterpret_cast<int16_t*>(compute_buffer.data());

  // compute the edit distance into the output column in-place
  thrust::transform(execpol->on(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings_count),
                    d_results,
                    edit_distance_levenshtein_algorithm{d_strings, d_targets, d_buffer, d_results});

  return results;
}

}  // namespace detail

// external APIs

/**
 * @copydoc nvtext::edit_distance
 */
std::unique_ptr<cudf::column> edit_distance(cudf::strings_column_view const& strings,
                                            cudf::strings_column_view const& targets,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::edit_distance(strings, targets, 0, mr);
}

}  // namespace nvtext
