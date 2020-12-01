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

#include <strings/utilities.cuh>

#include <nvtext/stemmer.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Return true if character at current iterator position
 * is a consonant.
 *
 * A consonant is a letter other than a, e, i, o or u, and other
 * than y preceded by a consonant.
 *
 * For `toy` the consonants are `t` and `y`, and in `syzygy` they
 * are `s`, `z` and `g`.
 *
 * A _vowel_ is defined as _not a consonant_.
 *
 * @param string_iterator Iterator positioned to the character to check.
 * @return True if the character at the iterator is a consonant.
 */
__device__ bool is_consonant(cudf::string_view::const_iterator string_iterator)
{
  auto ch = *string_iterator;
  cudf::string_view const d_vowels("aeiou", 5);
  if (d_vowels.find(ch) >= 0) return false;
  if ((ch != 'y') || (string_iterator.position() == 0)) return true;
  // for 'y' case, check previous character is a consonant
  --string_iterator;
  return d_vowels.find(*string_iterator) >= 0;
}

/**
 * @brief Functor for the detail::is_letter_fn called to return true/false
 * indicating the specified character is a consonant or a vowel.
 */
template <typename PositionIterator>
struct is_letter_fn {
  cudf::column_device_view const d_strings;
  letter_type ltype;
  PositionIterator position_itr;

  __device__ bool operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return false;
    auto const position = position_itr[idx];
    auto const length   = d_str.length();
    if ((position >= length) || (position < -length)) return false;
    return is_consonant(d_str.begin() + ((position + length) % length))
             ? ltype == letter_type::CONSONANT
             : ltype == letter_type::VOWEL;
  }
};

}  // namespace

// details API

template <typename PositionIterator>
std::unique_ptr<cudf::column> is_letter(cudf::strings_column_view const& strings,
                                        letter_type ltype,
                                        PositionIterator position_itr,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::BOOL8});

  // create empty output column
  auto results =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::BOOL8},
                                  strings.size(),
                                  cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                  strings.null_count(),
                                  stream,
                                  mr);
  // set values into output column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings.size()),
                    results->mutable_view().data<bool>(),
                    is_letter_fn<PositionIterator>{*strings_column, ltype, position_itr});
  return results;
}

namespace {

/**
 * @brief For dispatching index-type of indices parameter in the nvtext::is_letter API.
 */
struct dispatch_is_letter_fn {
  template <typename T, std::enable_if_t<cudf::is_index_type<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(cudf::strings_column_view const& strings,
                                           letter_type ltype,
                                           cudf::column_view const& indices,
                                           rmm::cuda_stream_view stream,
                                           rmm::mr::device_memory_resource* mr) const
  {
    CUDF_EXPECTS(strings.size() == indices.size(),
                 "strings column and indices column must be the same size");
    CUDF_EXPECTS(!indices.has_nulls(), "indices column must not contain nulls");
    // resolve and pass an iterator for the indices column to the detail function
    return is_letter(strings, ltype, indices.begin<T>(), stream, mr);
  }

  template <typename T, typename... Args, std::enable_if_t<not cudf::is_index_type<T>()>* = nullptr>
  std::unique_ptr<cudf::column> operator()(Args&&... args) const
  {
    CUDF_FAIL("The is_letter indices parameter must be an integer type.");
  }
};

/**
 * @brief Returns the measure for each string.
 *
 * Text description here is from https://tartarus.org/martin/PorterStemmer/def.txt
 *
 * A consonant will be denoted by `c`, a vowel by `v`. A list `ccc...` of length
 * greater than 0 will be denoted by `C`, and a list `vvv...` of length greater
 * than 0 will be denoted by `V`. Any word, or part of a word, therefore has one
 * of the four forms:
 *
 * @code{.pseudo}
 *     CVCV ... C
 *     CVCV ... V
 *     VCVC ... C
 *     VCVC ... V
 * @endcode
 *
 * These may all be represented by the single form `[C]VCVC ... [V]`
 * where the square brackets denote arbitrary presence of their contents.
 * Using `(VC){m}` to denote `VC` repeated `m` times, this may again be written as
 * `[C](VC){m}[V]`.
 *
 * And `m` will be called the _measure_ of any word or word part when represented in
 * this form. The case `m = 0` covers the null or empty string.
 *
 * Examples:
 * @code{.pseudo}
 * m=0:    TR,  EE,  TREE,  Y,  BY.
 * m=1:    TROUBLE,  OATS,  TREES,  IVY.
 * m=2:    TROUBLES,  PRIVATE,  OATEN,  ORRERY.
 * @endcode
 */
struct porter_stemmer_measure_fn {
  cudf::column_device_view const d_strings;  // strings to measure

  __device__ int32_t operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) return 0;
    cudf::string_view d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return 0;
    int32_t measure = 0;
    auto itr        = d_str.begin();
    bool vowel_run  = !is_consonant(itr);
    while (itr != d_str.end()) {
      if (is_consonant(itr)) {
        if (vowel_run) measure++;
        vowel_run = false;
      } else {
        vowel_run = true;
      }
      ++itr;
    }
    return measure;
  }
};

}  // namespace

std::unique_ptr<cudf::column> porter_stemmer_measure(cudf::strings_column_view const& strings,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32});

  // create empty output column
  auto results =
    cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                  strings.size(),
                                  cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                  strings.null_count(),
                                  stream,
                                  mr);
  // compute measures into output column
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(strings.size()),
                    results->mutable_view().data<int32_t>(),
                    porter_stemmer_measure_fn{*strings_column});
  return results;
}

std::unique_ptr<cudf::column> is_letter(cudf::strings_column_view const& strings,
                                        letter_type ltype,
                                        cudf::column_view const& indices,
                                        rmm::cuda_stream_view stream,
                                        rmm::mr::device_memory_resource* mr)
{
  return cudf::type_dispatcher(
    indices.type(), dispatch_is_letter_fn{}, strings, ltype, indices, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> is_letter(cudf::strings_column_view const& strings,
                                        letter_type ltype,
                                        cudf::size_type character_index,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_letter(strings,
                           ltype,
                           thrust::make_constant_iterator<cudf::size_type>(character_index),
                           rmm::cuda_stream_default,
                           mr);
}

std::unique_ptr<cudf::column> is_letter(cudf::strings_column_view const& strings,
                                        letter_type ltype,
                                        cudf::column_view const& indices,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_letter(strings, ltype, indices, rmm::cuda_stream_default, mr);
}

/**
 * @copydoc nvtext::porter_stemmer_measure
 */
std::unique_ptr<cudf::column> porter_stemmer_measure(cudf::strings_column_view const& strings,
                                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::porter_stemmer_measure(strings, rmm::cuda_stream_default, mr);
}

}  // namespace nvtext
