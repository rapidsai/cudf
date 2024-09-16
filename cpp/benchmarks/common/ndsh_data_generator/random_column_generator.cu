/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "random_column_generator.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/filling.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <string>

namespace cudf::datagen {

namespace {

// Functor for generating random strings
struct random_string_generator {
  char* chars;
  cudf::string_view corpus;

  CUDF_HOST_DEVICE random_string_generator(char* c, cudf::string_view crps) : chars(c), corpus(crps)
  {
  }

  __device__ void operator()(thrust::tuple<int64_t, int64_t> str_begin_end)
  {
    // Get the begin and end offsets
    auto begin = thrust::get<0>(str_begin_end);
    auto end   = thrust::get<1>(str_begin_end);

    // Define the thrust random engine
    thrust::default_random_engine engine;
    thrust::uniform_int_distribution<int> dist(0, 8000);
    engine.discard(begin);
    engine.discard(end);

    // Generate a random offset
    auto offset = dist(engine);
    for (auto idx = begin; idx < end; idx++) {
      chars[idx] = corpus[offset];
      offset += 1;
    }
  }
};

// Functor for generating random numbers
template <typename T>
struct random_number_generator {
  T lower;
  T upper;

  CUDF_HOST_DEVICE random_number_generator(T lower, T upper) : lower(lower), upper(upper) {}

  __device__ T operator()(const int64_t idx) const
  {
    if constexpr (cudf::is_integral<T>()) {
      thrust::default_random_engine engine;
      thrust::uniform_int_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    } else {
      thrust::default_random_engine engine;
      thrust::uniform_real_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    }
  }
};

}  // namespace

std::unique_ptr<cudf::column> generate_verb_phrase(cudf::size_type num_rows)
{
  CUDF_FUNC_RANGE();
  constexpr std::array verbs = {"sleep",
                                "wake",
                                "are",
                                "cajole"
                                "haggle",
                                "nag",
                                "use",
                                "boost",
                                "affix",
                                "detect",
                                "integrate",
                                "maintain"
                                "nod",
                                "was",
                                "lose",
                                "sublate",
                                "solve",
                                "thrash",
                                "promise",
                                "engage",
                                "hinder",
                                "print",
                                "x-ray",
                                "breach",
                                "eat",
                                "grow",
                                "impress",
                                "mold",
                                "poach",
                                "serve",
                                "run",
                                "dazzle",
                                "snooze",
                                "doze",
                                "unwind",
                                "kindle",
                                "play",
                                "hang",
                                "believe",
                                "doubt"};
  return generate_random_string_column_from_set(
    cudf::host_span<const char* const>(verbs.data(), verbs.size()), num_rows);
}

std::unique_ptr<cudf::column> generate_noun_phrase(cudf::size_type num_rows)
{
  CUDF_FUNC_RANGE();
  constexpr std::array nouns = {"foxes",
                                "ideas",
                                "theodolites",
                                "pinto beans",
                                "instructions",
                                "dependencies",
                                "excuses",
                                "platelets",
                                "asymptotes",
                                "courts",
                                "dolphins",
                                "multipliers",
                                "sauternes",
                                "warthogs",
                                "frets",
                                "dinos",
                                "attainments",
                                "somas",
                                "Tiresias' patterns",
                                "forges",
                                "braids",
                                "hockey players",
                                "frays",
                                "warhorses",
                                "dugouts",
                                "notornis",
                                "epitaphs",
                                "pearls",
                                "tithes",
                                "waters",
                                "orbits",
                                "gifts",
                                "sheaves",
                                "depths",
                                "sentiments",
                                "decoys",
                                "realms",
                                "pains",
                                "grouches",
                                "escapades"};
  return generate_random_string_column_from_set(
    cudf::host_span<const char* const>(nouns.data(), nouns.size()), num_rows);
}

std::unique_ptr<cudf::column> generate_terminator(cudf::size_type num_rows)
{
  CUDF_FUNC_RANGE();
  constexpr std::array terminators = {".", ";", ":", "?", "!", "--"};
  return generate_random_string_column_from_set(
    cudf::host_span<const char* const>(terminators.data(), terminators.size()), num_rows);
}

std::unique_ptr<cudf::column> generate_sentence(cudf::size_type num_rows)
{
  CUDF_FUNC_RANGE();
  auto const verb_phrase = generate_verb_phrase(num_rows);
  auto const noun_phrase = generate_noun_phrase(num_rows);
  auto const terminator  = generate_terminator(num_rows);
  auto const sentence_parts =
    cudf::table_view({verb_phrase->view(), noun_phrase->view(), terminator->view()});
  auto const sentence = cudf::strings::concatenate(sentence_parts,
                                                   cudf::string_scalar(""),
                                                   cudf::string_scalar("", false),
                                                   cudf::strings::separator_on_nulls::NO);
  return cudf::strings::join_strings(sentence->view());
}

cudf::string_view generate_text_corpus()
{
  CUDF_FUNC_RANGE();
  constexpr cudf::size_type num_rows       = 10'000;
  auto text_corpus_column                  = generate_sentence(num_rows);
  auto text_corpus_contents                = text_corpus_column->release();
  std::unique_ptr<rmm::device_buffer> buff = std::move(text_corpus_contents.data);
  return cudf::string_view((char*)buff->data(), buff->size());
}

std::unique_ptr<cudf::column> generate_random_string_column(cudf::size_type lower,
                                                            cudf::size_type upper,
                                                            cudf::size_type num_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const text_corpus = generate_text_corpus();

  auto offsets_begin = cudf::detail::make_counting_transform_iterator(
    0, random_number_generator<cudf::size_type>(lower, upper));
  auto [offsets_column, computed_bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_begin, offsets_begin + num_rows, stream, mr);
  rmm::device_uvector<char> chars(computed_bytes, stream);

  auto const offset_itr =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // We generate the strings in parallel into the `chars` vector using the
  // offsets vector generated above.
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_zip_iterator(offset_itr, offset_itr + 1),
                     num_rows,
                     random_string_generator(chars.data(), text_corpus));

  return cudf::make_strings_column(
    num_rows, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

template <typename T>
std::unique_ptr<cudf::column> generate_random_numeric_column(T lower,
                                                             T upper,
                                                             cudf::size_type num_rows,
                                                             rmm::cuda_stream_view stream,
                                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::size_type begin = 0;
  cudf::size_type end   = num_rows;
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(begin),
                    thrust::make_counting_iterator(end),
                    col->mutable_view().begin<T>(),
                    random_number_generator<T>(lower, upper));
  return col;
}

template std::unique_ptr<cudf::column> generate_random_numeric_column<int8_t>(
  int8_t lower,
  int8_t upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<int16_t>(
  int16_t lower,
  int16_t upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<cudf::size_type>(
  cudf::size_type lower,
  cudf::size_type upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_random_numeric_column<double>(
  double lower,
  double upper,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> generate_primary_key_column(cudf::scalar const& start,
                                                          cudf::size_type num_rows,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::sequence(num_rows, start, stream, mr);
}

std::unique_ptr<cudf::column> generate_repeat_string_column(std::string const& value,
                                                            cudf::size_type num_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const scalar = cudf::string_scalar(value);
  return cudf::make_column_from_scalar(scalar, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> generate_random_string_column_from_set(
  cudf::host_span<const char* const> set,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Build a gather map of random strings to choose from
  // The size of the string sets always fits within 16-bit integers
  auto const indices =
    generate_primary_key_column(cudf::numeric_scalar<int16_t>(0), set.size(), stream, mr);
  auto const keys       = cudf::test::strings_column_wrapper(set.begin(), set.end()).release();
  auto const gather_map = cudf::table_view({indices->view(), keys->view()});

  // Build a column of random keys to gather from the set
  auto const gather_keys =
    generate_random_numeric_column<int16_t>(0, set.size() - 1, num_rows, stream, mr);

  // Perform the gather operation
  auto const gathered_table = cudf::gather(
    gather_map, gather_keys->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto gathered_table_columns = gathered_table->release();
  return std::move(gathered_table_columns[1]);
}

template <typename T>
std::unique_ptr<cudf::column> generate_repeat_sequence_column(T seq_length,
                                                              bool zero_indexed,
                                                              cudf::size_type num_rows,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto pkey =
    generate_primary_key_column(cudf::numeric_scalar<cudf::size_type>(0), num_rows, stream, mr);
  auto repeat_seq_zero_indexed = cudf::binary_operation(pkey->view(),
                                                        cudf::numeric_scalar<T>(seq_length),
                                                        cudf::binary_operator::MOD,
                                                        cudf::data_type{cudf::type_to_id<T>()},
                                                        stream,
                                                        mr);
  if (zero_indexed) { return repeat_seq_zero_indexed; }
  return cudf::binary_operation(repeat_seq_zero_indexed->view(),
                                cudf::numeric_scalar<T>(1),
                                cudf::binary_operator::ADD,
                                cudf::data_type{cudf::type_to_id<T>()},
                                stream,
                                mr);
}

template std::unique_ptr<cudf::column> generate_repeat_sequence_column<int8_t>(
  int8_t seq_length,
  bool zero_indexed,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> generate_repeat_sequence_column<cudf::size_type>(
  cudf::size_type seq_length,
  bool zero_indexed,
  cudf::size_type num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::datagen
