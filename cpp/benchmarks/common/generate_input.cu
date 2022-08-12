/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "generate_input.hpp"
#include "random_distribution_factory.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/filling.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

/**
 * @brief Mersenne Twister pseudo-random engine.
 */
auto deterministic_engine(unsigned seed) { return thrust::minstd_rand{seed}; }

/**
 *  Computes the mean value for a distribution of given type and value bounds.
 */
template <typename T>
T get_distribution_mean(distribution_params<T> const& dist)
{
  switch (dist.id) {
    case distribution_id::NORMAL:
    case distribution_id::UNIFORM: return (dist.lower_bound / 2.) + (dist.upper_bound / 2.);
    case distribution_id::GEOMETRIC: {
      auto const range_size = dist.lower_bound < dist.upper_bound
                                ? dist.upper_bound - dist.lower_bound
                                : dist.lower_bound - dist.upper_bound;
      auto const p          = geometric_dist_p(range_size);
      if (dist.lower_bound < dist.upper_bound)
        return dist.lower_bound + (1. / p);
      else
        return dist.lower_bound - (1. / p);
    }
    default: CUDF_FAIL("Unsupported distribution type.");
  }
}

/**
 * @brief Computes the average element size in a column, given the data profile.
 *
 * Random distribution parameters like average string length and maximum list nesting level affect
 * the element size of non-fixed-width columns. For lists and structs, `avg_element_size` is called
 * recursively to determine the size of nested columns.
 */
size_t avg_element_size(data_profile const& profile, cudf::data_type dtype);

// Utilities to determine the mean size of an element, given the data profile
template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_width<T>())>
size_t non_fixed_width_size(data_profile const& profile)
{
  CUDF_FAIL("Should not be called, use `size_of` for this type instead");
}

template <typename T, CUDF_ENABLE_IF(!cudf::is_fixed_width<T>())>
size_t non_fixed_width_size(data_profile const& profile)
{
  CUDF_FAIL("not implemented!");
}

template <>
size_t non_fixed_width_size<cudf::string_view>(data_profile const& profile)
{
  auto const dist = profile.get_distribution_params<cudf::string_view>().length_params;
  return get_distribution_mean(dist);
}

template <>
size_t non_fixed_width_size<cudf::list_view>(data_profile const& profile)
{
  auto const dist_params       = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean = get_distribution_mean(dist_params.length_params);
  auto const element_size = avg_element_size(profile, cudf::data_type{dist_params.element_type});
  return element_size * pow(single_level_mean, dist_params.max_depth);
}

template <>
size_t non_fixed_width_size<cudf::struct_view>(data_profile const& profile)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();
  return std::accumulate(dist_params.leaf_types.cbegin(),
                         dist_params.leaf_types.cend(),
                         0ul,
                         [&](auto& sum, auto type_id) {
                           return sum + avg_element_size(profile, cudf::data_type{type_id});
                         });
}

struct non_fixed_width_size_fn {
  template <typename T>
  size_t operator()(data_profile const& profile)
  {
    return non_fixed_width_size<T>(profile);
  }
};

size_t avg_element_size(data_profile const& profile, cudf::data_type dtype)
{
  if (cudf::is_fixed_width(dtype)) { return cudf::size_of(dtype); }
  return cudf::type_dispatcher(dtype, non_fixed_width_size_fn{}, profile);
}

/**
 * @brief bool generator with given probability [0.0 - 1.0] of returning true.
 */
struct bool_generator {
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> dist;
  double probability_true;
  bool_generator(thrust::minstd_rand engine, double probability_true)
    : engine(engine), dist{0, 1}, probability_true{probability_true}
  {
  }
  bool_generator(unsigned seed, double probability_true)
    : engine(seed), dist{0, 1}, probability_true{probability_true}
  {
  }

  __device__ bool operator()(size_t n)
  {
    engine.discard(n);
    return dist(engine) < probability_true;
  }
};

/**
 * @brief Functor that computes a random column element with the given data profile.
 *
 * The implementation is SFINAEd for different type groups. Currently only used for fixed-width
 * types.
 */
template <typename T, typename Enable = void>
struct random_value_fn;

/**
 * @brief Creates an random timestamp/duration value
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<cudf::is_chrono<T>()>> {
  distribution_fn<int64_t> seconds_gen;
  distribution_fn<int64_t> nanoseconds_gen;

  random_value_fn(distribution_params<T> params)
  {
    using cuda::std::chrono::duration_cast;

    std::pair<cudf::duration_s, cudf::duration_s> const range_s = {
      duration_cast<cuda::std::chrono::seconds>(typename T::duration{params.lower_bound}),
      duration_cast<cuda::std::chrono::seconds>(typename T::duration{params.upper_bound})};
    if (range_s.first != range_s.second) {
      seconds_gen =
        make_distribution<int64_t>(params.id, range_s.first.count(), range_s.second.count());

      nanoseconds_gen = make_distribution<int64_t>(distribution_id::UNIFORM, 0l, 1000000000l);
    } else {
      // Don't need a random seconds generator for sub-second intervals
      seconds_gen = [range_s](thrust::minstd_rand&, size_t size) {
        rmm::device_uvector<int64_t> result(size, cudf::default_stream_value);
        thrust::fill(thrust::device, result.begin(), result.end(), range_s.second.count());
        return result;
      };

      std::pair<cudf::duration_ns, cudf::duration_ns> const range_ns = {
        duration_cast<cudf::duration_ns>(typename T::duration{params.lower_bound}),
        duration_cast<cudf::duration_ns>(typename T::duration{params.upper_bound})};
      nanoseconds_gen = make_distribution<int64_t>(distribution_id::UNIFORM,
                                                   std::min(range_ns.first.count(), 0l),
                                                   std::max(range_ns.second.count(), 0l));
    }
  }

  rmm::device_uvector<T> operator()(thrust::minstd_rand& engine, unsigned size)
  {
    auto const sec = seconds_gen(engine, size);
    auto const ns  = nanoseconds_gen(engine, size);
    rmm::device_uvector<T> result(size, cudf::default_stream_value);
    thrust::transform(
      thrust::device,
      sec.begin(),
      sec.end(),
      ns.begin(),
      result.begin(),
      [] __device__(int64_t sec_value, int64_t nanoseconds_value) {
        auto const timestamp_ns =
          cudf::duration_s{sec_value} + cudf::duration_ns{nanoseconds_value};
        // Return value in the type's precision
        return T(cuda::std::chrono::duration_cast<typename T::duration>(timestamp_ns));
      });
    return result;
  }
};

/**
 * @brief Creates an random fixed_point value.
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<cudf::is_fixed_point<T>()>> {
  using rep = typename T::rep;
  rep const lower_bound;
  rep const upper_bound;
  distribution_fn<rep> dist;
  std::optional<numeric::scale_type> scale;

  random_value_fn(distribution_params<rep> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<rep>(desc.id, desc.lower_bound, desc.upper_bound)}
  {
  }

  rmm::device_uvector<T> operator()(thrust::minstd_rand& engine, unsigned size)
  {
    if (not scale.has_value()) {
      int const max_scale = std::numeric_limits<rep>::digits10;
      std::uniform_int_distribution<int> scale_dist{-max_scale, max_scale};
      std::mt19937 engine_scale(engine());
      scale = numeric::scale_type{scale_dist(engine_scale)};
    }
    auto const ints = dist(engine, size);
    rmm::device_uvector<T> result(size, cudf::default_stream_value);
    // Clamp the generated random value to the specified range
    thrust::transform(thrust::device,
                      ints.begin(),
                      ints.end(),
                      result.begin(),
                      [scale       = *(this->scale),
                       upper_bound = this->upper_bound,
                       lower_bound = this->lower_bound] __device__(auto int_value) {
                        return T{std::clamp(int_value, lower_bound, upper_bound), scale};
                      });
    return result;
  }
};

/**
 * @brief Creates an random numeric value with the given distribution.
 */
template <typename T>
struct random_value_fn<T, std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>> {
  T const lower_bound;
  T const upper_bound;
  distribution_fn<T> dist;

  random_value_fn(distribution_params<T> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<T>(desc.id, desc.lower_bound, desc.upper_bound)}
  {
  }

  auto operator()(thrust::minstd_rand& engine, unsigned size) { return dist(engine, size); }
};

/**
 * @brief Creates an boolean value with given probability of returning `true`.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<std::is_same_v<T, bool>>> {
  // Bernoulli distribution
  distribution_fn<bool> dist;

  random_value_fn(distribution_params<bool> const& desc)
    : dist{[valid_prob = desc.probability_true](thrust::minstd_rand& engine,
                                                size_t size) -> rmm::device_uvector<bool> {
        rmm::device_uvector<bool> result(size, cudf::default_stream_value);
        thrust::tabulate(
          thrust::device, result.begin(), result.end(), bool_generator(engine, valid_prob));
        return result;
      }}
  {
  }
  auto operator()(thrust::minstd_rand& engine, unsigned size) { return dist(engine, size); }
};

auto create_run_length_dist(cudf::size_type avg_run_len)
{
  // Distribution with low probability of generating 0-1 even with a low `avg_run_len` value
  static constexpr float alpha = 4.f;
  return std::gamma_distribution<float>{alpha, avg_run_len / alpha};
}

/**
 * @brief Generate indices within range [0 , cardinality) repeating with average run length
 * `avg_run_len`
 *
 * @param avg_run_len  Average run length of the generated indices
 * @param cardinality  Number of unique values in the output vector
 * @param num_rows     Number of indices to generate
 * @param engine       Random engine
 * @return Generated indices of type `cudf::size_type`
 */
rmm::device_uvector<cudf::size_type> sample_indices_with_run_length(cudf::size_type avg_run_len,
                                                                    cudf::size_type cardinality,
                                                                    cudf::size_type num_rows,
                                                                    thrust::minstd_rand& engine)
{
  auto sample_dist = random_value_fn<cudf::size_type>{
    distribution_params<cudf::size_type>{distribution_id::UNIFORM, 0, cardinality - 1}};
  if (avg_run_len > 1) {
    auto avglen_dist =
      random_value_fn<int>{distribution_params<int>{distribution_id::UNIFORM, 1, 2 * avg_run_len}};
    auto const approx_run_len = num_rows / avg_run_len + 1;
    auto run_lens             = avglen_dist(engine, approx_run_len);
    thrust::inclusive_scan(
      thrust::device, run_lens.begin(), run_lens.end(), run_lens.begin(), std::plus<int>{});
    auto const samples_indices = sample_dist(engine, approx_run_len + 1);
    // This is gather.
    auto avg_repeated_sample_indices_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [rb              = run_lens.begin(),
       re              = run_lens.end(),
       samples_indices = samples_indices.begin()] __device__(cudf::size_type i) {
        auto sample_idx = thrust::upper_bound(thrust::seq, rb, re, i) - rb;
        return samples_indices[sample_idx];
      });
    rmm::device_uvector<cudf::size_type> repeated_sample_indices(num_rows,
                                                                 cudf::default_stream_value);
    thrust::copy(thrust::device,
                 avg_repeated_sample_indices_iterator,
                 avg_repeated_sample_indices_iterator + num_rows,
                 repeated_sample_indices.begin());
    return repeated_sample_indices;
  } else {
    // generate n samples.
    return sample_dist(engine, num_rows);
  }
}

/**
 * @brief Creates a column with random content of type @ref T.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @tparam T Data type of the output column
 * @return Column filled with random data
 */
template <typename T>
std::unique_ptr<cudf::column> create_random_column(data_profile const& profile,
                                                   thrust::minstd_rand& engine,
                                                   cudf::size_type num_rows)
{
  // Bernoulli distribution
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});
  auto value_dist = random_value_fn<T>{profile.get_distribution_params<T>()};

  // Distribution for picking elements from the array of samples
  auto const avg_run_len = profile.get_avg_run_length();
  rmm::device_uvector<T> data(0, cudf::default_stream_value);
  rmm::device_uvector<bool> null_mask(0, cudf::default_stream_value);

  if (profile.get_cardinality() == 0 and avg_run_len == 1) {
    data      = value_dist(engine, num_rows);
    null_mask = valid_dist(engine, num_rows);
  } else {
    auto const cardinality = [profile_cardinality = profile.get_cardinality(), num_rows] {
      return (profile_cardinality == 0 or profile_cardinality > num_rows) ? num_rows
                                                                          : profile_cardinality;
    }();
    rmm::device_uvector<bool> samples_null_mask = valid_dist(engine, cardinality);
    rmm::device_uvector<T> samples              = value_dist(engine, cardinality);
    // generate n samples and gather.
    auto const sample_indices =
      sample_indices_with_run_length(avg_run_len, cardinality, num_rows, engine);
    data      = rmm::device_uvector<T>(num_rows, cudf::default_stream_value);
    null_mask = rmm::device_uvector<bool>(num_rows, cudf::default_stream_value);
    thrust::gather(
      thrust::device, sample_indices.begin(), sample_indices.end(), samples.begin(), data.begin());
    thrust::gather(thrust::device,
                   sample_indices.begin(),
                   sample_indices.end(),
                   samples_null_mask.begin(),
                   null_mask.begin());
  }

  auto [result_bitmask, null_count] =
    cudf::detail::valid_if(null_mask.begin(), null_mask.end(), thrust::identity<bool>{});

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_to_id<T>()},
    num_rows,
    data.release(),
    profile.get_null_probability().has_value() ? std::move(result_bitmask) : rmm::device_buffer{});
}

struct valid_or_zero {
  template <typename T>
  __device__ T operator()(thrust::tuple<T, bool> len_valid) const
  {
    return thrust::get<1>(len_valid) ? thrust::get<0>(len_valid) : T{0};
  }
};

struct string_generator {
  char* chars;
  thrust::minstd_rand engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;
  string_generator(char* c, thrust::minstd_rand& engine)
    : chars(c), engine(engine), char_dist(32, 137)
  // ~90% ASCII, ~10% UTF-8.
  // ~80% not-space, ~20% space.
  // range 32-127 is ASCII; 127-136 will be multi-byte UTF-8
  {
  }
  __device__ void operator()(thrust::tuple<cudf::size_type, cudf::size_type> str_begin_end)
  {
    auto begin = thrust::get<0>(str_begin_end);
    auto end   = thrust::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      auto ch = char_dist(engine);
      if (i == end - 1 && ch >= '\x7F') ch = ' ';  // last element ASCII only.
      if (ch >= '\x7F')                            // x7F is at the top edge of ASCII
        chars[i++] = '\xC4';                       // these characters are assigned two bytes
      chars[i] = static_cast<char>(ch + (ch >= '\x7F'));
    }
  }
};

/**
 * @brief Create a UTF-8 string column with the average length.
 *
 */
std::unique_ptr<cudf::column> create_random_utf8_string_column(data_profile const& profile,
                                                               thrust::minstd_rand& engine,
                                                               cudf::size_type num_rows)
{
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::string_view>().length_params};
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});
  auto lengths   = len_dist(engine, num_rows + 1);
  auto null_mask = valid_dist(engine, num_rows + 1);
  thrust::transform_if(
    thrust::device,
    lengths.begin(),
    lengths.end(),
    null_mask.begin(),
    lengths.begin(),
    [] __device__(auto) { return 0; },
    thrust::logical_not<bool>{});
  auto valid_lengths = thrust::make_transform_iterator(
    thrust::make_zip_iterator(thrust::make_tuple(lengths.begin(), null_mask.begin())),
    valid_or_zero{});
  rmm::device_uvector<cudf::size_type> offsets(num_rows + 1, cudf::default_stream_value);
  thrust::exclusive_scan(
    thrust::device, valid_lengths, valid_lengths + lengths.size(), offsets.begin());
  // offfsets are ready.
  auto chars_length = *thrust::device_pointer_cast(offsets.end() - 1);
  rmm::device_uvector<char> chars(chars_length, cudf::default_stream_value);
  thrust::for_each_n(thrust::device,
                     thrust::make_zip_iterator(offsets.begin(), offsets.begin() + 1),
                     num_rows,
                     string_generator{chars.data(), engine});
  auto [result_bitmask, null_count] =
    cudf::detail::valid_if(null_mask.begin(), null_mask.end() - 1, thrust::identity<bool>{});
  return cudf::make_strings_column(
    num_rows,
    std::move(offsets),
    std::move(chars),
    profile.get_null_probability().has_value() ? std::move(result_bitmask) : rmm::device_buffer{});
}

/**
 * @brief Creates a string column with random content.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @return Column filled with random strings
 */
template <>
std::unique_ptr<cudf::column> create_random_column<cudf::string_view>(data_profile const& profile,
                                                                      thrust::minstd_rand& engine,
                                                                      cudf::size_type num_rows)
{
  auto const cardinality = std::min(profile.get_cardinality(), num_rows);
  auto const avg_run_len = profile.get_avg_run_length();

  auto sample_strings =
    create_random_utf8_string_column(profile, engine, cardinality == 0 ? num_rows : cardinality);
  if (cardinality == 0) { return sample_strings; }
  auto sample_indices = sample_indices_with_run_length(avg_run_len, cardinality, num_rows, engine);
  auto str_table      = cudf::detail::gather(cudf::table_view{{sample_strings->view()}},
                                        sample_indices,
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        cudf::detail::negative_index_policy::NOT_ALLOWED);
  return std::move(str_table->release()[0]);
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::dictionary32>(data_profile const& profile,
                                                                       thrust::minstd_rand& engine,
                                                                       cudf::size_type num_rows)
{
  CUDF_FAIL("not implemented yet");
}

/**
 * @brief Functor to dispatch create_random_column calls.
 */
struct create_rand_col_fn {
 public:
  template <typename T>
  std::unique_ptr<cudf::column> operator()(data_profile const& profile,
                                           thrust::minstd_rand& engine,
                                           cudf::size_type num_rows)
  {
    return create_random_column<T>(profile, engine, num_rows);
  }
};

/**
 * @brief Calculates the number of direct parents needed to generate a struct column hierarchy with
 * lowest maximum number of children in any nested column.
 *
 * Used to generate an "evenly distributed" struct column hierarchy with the given number of leaf
 * columns and nesting levels. The column tree is considered evenly distributed if all columns have
 * nearly the same number of child columns (difference not larger than one).
 */
int num_direct_parents(int num_lvls, int num_leaf_columns)
{
  // Estimated average number of children in the hierarchy;
  auto const num_children_avg = std::pow(num_leaf_columns, 1. / num_lvls);
  // Minimum number of children columns for any column in the hierarchy
  int const num_children_min = std::floor(num_children_avg);
  // Maximum number of children columns for any column in the hierarchy
  int const num_children_max = num_children_min + 1;

  // Minimum number of columns needed so that their number of children does not exceed the maximum
  int const min_for_current_nesting = std::ceil((double)num_leaf_columns / num_children_max);
  // Minimum number of columns needed so that columns at the higher levels have at least the minimum
  // number of children
  int const min_for_upper_nesting = std::pow(num_children_min, num_lvls - 1);
  // Both conditions need to be satisfied
  return std::max(min_for_current_nesting, min_for_upper_nesting);
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::struct_view>(data_profile const& profile,
                                                                      thrust::minstd_rand& engine,
                                                                      cudf::size_type num_rows)
{
  auto const dist_params = profile.get_distribution_params<cudf::struct_view>();

  // Generate leaf columns
  std::vector<std::unique_ptr<cudf::column>> children;
  children.reserve(dist_params.leaf_types.size());
  std::transform(dist_params.leaf_types.cbegin(),
                 dist_params.leaf_types.cend(),
                 std::back_inserter(children),
                 [&](auto& type_id) {
                   return cudf::type_dispatcher(
                     cudf::data_type(type_id), create_rand_col_fn{}, profile, engine, num_rows);
                 });

  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});

  // Generate the column bottom-up
  for (int lvl = dist_params.max_depth; lvl > 0; --lvl) {
    // Generating the next level
    std::vector<std::unique_ptr<cudf::column>> parents;
    parents.resize(num_direct_parents(lvl, children.size()));

    auto current_child = children.begin();
    for (auto current_parent = parents.begin(); current_parent != parents.end(); ++current_parent) {
      auto [null_mask, null_count] = [&]() {
        if (profile.get_null_probability().has_value()) {
          auto valids = valid_dist(engine, num_rows);
          return cudf::detail::valid_if(valids.begin(), valids.end(), thrust::identity<bool>{});
        }
        return std::pair<rmm::device_buffer, cudf::size_type>{};
      }();

      // Adopt remaining children as evenly as possible
      auto const num_to_adopt = cudf::util::div_rounding_up_unsafe(
        std::distance(current_child, children.end()), std::distance(current_parent, parents.end()));
      CUDF_EXPECTS(num_to_adopt > 0, "No children columns left to adopt");

      std::vector<std::unique_ptr<cudf::column>> children_to_adopt;
      children_to_adopt.insert(children_to_adopt.end(),
                               std::make_move_iterator(current_child),
                               std::make_move_iterator(current_child + num_to_adopt));
      current_child += children_to_adopt.size();

      *current_parent = cudf::make_structs_column(
        num_rows, std::move(children_to_adopt), null_count, std::move(null_mask));
    }

    if (lvl == 1) {
      CUDF_EXPECTS(parents.size() == 1, "There should be one top-level column");
      return std::move(parents.front());
    }
    children = std::move(parents);
  }
  CUDF_FAIL("Reached unreachable code in struct column creation");
}

template <typename T>
struct clamp_down : public thrust::unary_function<T, T> {
  T max;
  clamp_down(T max) : max(max) {}
  __host__ __device__ T operator()(T x) const { return min(x, max); }
};
/**
 * @brief Creates a list column with random content.
 *
 * The data profile determines the list length distribution, number of nested level, and the data
 * type of the bottom level.
 *
 * @param profile Parameters for the random generator
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output column
 *
 * @return Column filled with random lists
 */
template <>
std::unique_ptr<cudf::column> create_random_column<cudf::list_view>(data_profile const& profile,
                                                                    thrust::minstd_rand& engine,
                                                                    cudf::size_type num_rows)
{
  auto const dist_params       = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean = get_distribution_mean(dist_params.length_params);
  auto const num_elements      = num_rows * pow(single_level_mean, dist_params.max_depth);

  auto leaf_column = cudf::type_dispatcher(
    cudf::data_type(dist_params.element_type), create_rand_col_fn{}, profile, engine, num_elements);
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::list_view>().length_params};
  auto valid_dist = random_value_fn<bool>(
    distribution_params<bool>{1. - profile.get_null_probability().value_or(0)});

  // Generate the list column bottom-up
  auto list_column = std::move(leaf_column);
  for (int lvl = 0; lvl < dist_params.max_depth; ++lvl) {
    // Generating the next level - offsets point into the current list column
    auto current_child_column      = std::move(list_column);
    cudf::size_type const num_rows = current_child_column->size() / single_level_mean;

    auto offsets = len_dist(engine, num_rows + 1);
    auto valids  = valid_dist(engine, num_rows);
    // to ensure these values <= current_child_column->size()
    auto output_offsets = thrust::make_transform_output_iterator(
      offsets.begin(), clamp_down{current_child_column->size()});

    thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.end(), output_offsets);
    thrust::device_pointer_cast(offsets.end())[-1] =
      current_child_column->size();  // Always include all elements

    auto offsets_column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release());

    auto [null_mask, null_count] =
      cudf::detail::valid_if(valids.begin(), valids.end(), thrust::identity<bool>{});
    list_column = cudf::make_lists_column(
      num_rows,
      std::move(offsets_column),
      std::move(current_child_column),
      profile.get_null_probability().has_value() ? null_count : 0,  // cudf::UNKNOWN_NULL_COUNT,
      profile.get_null_probability().has_value() ? std::move(null_mask) : rmm::device_buffer{});
  }
  return list_column;  // return the top-level column
}

using columns_vector = std::vector<std::unique_ptr<cudf::column>>;

/**
 * @brief Creates a vector of columns with random content.
 *
 * @param profile Parameters for the random generator
 * @param dtype_ids vector of data type ids, one for each output column
 * @param engine Pseudo-random engine
 * @param num_rows Size of the output columns
 *
 * @return Column filled with random lists
 */
columns_vector create_random_columns(data_profile const& profile,
                                     std::vector<cudf::type_id> dtype_ids,
                                     thrust::minstd_rand engine,
                                     cudf::size_type num_rows)
{
  columns_vector output_columns;
  std::transform(
    dtype_ids.begin(), dtype_ids.end(), std::back_inserter(output_columns), [&](auto tid) {
      engine.discard(num_rows);
      return cudf::type_dispatcher(
        cudf::data_type(tid), create_rand_col_fn{}, profile, engine, num_rows);
    });
  return output_columns;
}

/**
 * @brief Repeats the input data types cyclically order to fill a vector of @ref num_cols
 * elements.
 */
std::vector<cudf::type_id> cycle_dtypes(std::vector<cudf::type_id> const& dtype_ids,
                                        cudf::size_type num_cols)
{
  if (dtype_ids.size() == static_cast<std::size_t>(num_cols)) { return dtype_ids; }
  std::vector<cudf::type_id> out_dtypes;
  out_dtypes.reserve(num_cols);
  for (cudf::size_type col = 0; col < num_cols; ++col)
    out_dtypes.push_back(dtype_ids[col % dtype_ids.size()]);
  return out_dtypes;
}

std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 table_size_bytes table_bytes,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  size_t const avg_row_bytes =
    std::accumulate(dtype_ids.begin(), dtype_ids.end(), 0ul, [&](size_t sum, auto tid) {
      return sum + avg_element_size(profile, cudf::data_type(tid));
    });
  cudf::size_type const num_rows = table_bytes.size / avg_row_bytes;

  return create_random_table(dtype_ids, row_count{num_rows}, profile, seed);
}

std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 row_count num_rows,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  thrust::uniform_int_distribution<unsigned> seed_dist;

  columns_vector output_columns;
  std::transform(
    dtype_ids.begin(), dtype_ids.end(), std::back_inserter(output_columns), [&](auto tid) mutable {
      auto engine = deterministic_engine(seed_dist(seed_engine));
      return cudf::type_dispatcher(
        cudf::data_type(tid), create_rand_col_fn{}, profile, engine, num_rows.count);
    });
  return std::make_unique<cudf::table>(std::move(output_columns));
}

std::unique_ptr<cudf::table> create_sequence_table(std::vector<cudf::type_id> const& dtype_ids,
                                                   row_count num_rows,
                                                   std::optional<double> null_probability,
                                                   unsigned seed)
{
  auto seed_engine = deterministic_engine(seed);
  thrust::uniform_int_distribution<unsigned> seed_dist;

  auto columns = std::vector<std::unique_ptr<cudf::column>>(dtype_ids.size());
  std::transform(dtype_ids.begin(), dtype_ids.end(), columns.begin(), [&](auto dtype) mutable {
    auto init = cudf::make_default_constructed_scalar(cudf::data_type{dtype});
    auto col  = cudf::sequence(num_rows.count, *init);
    auto [mask, count] =
      create_random_null_mask(num_rows.count, null_probability, seed_dist(seed_engine));
    col->set_null_mask(std::move(mask), count);
    return col;
  });
  return std::make_unique<cudf::table>(std::move(columns));
}

std::pair<rmm::device_buffer, cudf::size_type> create_random_null_mask(
  cudf::size_type size, std::optional<double> null_probability, unsigned seed)
{
  if (not null_probability.has_value()) { return {rmm::device_buffer{}, 0}; }
  CUDF_EXPECTS(*null_probability >= 0.0 and *null_probability <= 1.0,
               "Null probability must be within the range [0.0, 1.0]");
  if (*null_probability == 0.0f) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_VALID), 0};
  } else if (*null_probability == 1.0) {
    return {cudf::create_null_mask(size, cudf::mask_state::ALL_NULL), size};
  } else {
    return cudf::detail::valid_if(thrust::make_counting_iterator<cudf::size_type>(0),
                                  thrust::make_counting_iterator<cudf::size_type>(size),
                                  bool_generator{seed, 1.0 - *null_probability});
  }
}

std::vector<cudf::type_id> get_type_or_group(int32_t id)
{
  // identity transformation when passing a concrete type_id
  if (id < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS))
    return {static_cast<cudf::type_id>(id)};

  // if the value is larger that type_id::NUM_TYPE_IDS, it's a group id
  type_group_id const group_id = static_cast<type_group_id>(id);

  using trait_fn       = bool (*)(cudf::data_type);
  trait_fn is_integral = [](cudf::data_type type) {
    return cudf::is_numeric(type) && !cudf::is_floating_point(type);
  };
  trait_fn is_integral_signed = [](cudf::data_type type) {
    return cudf::is_numeric(type) && !cudf::is_floating_point(type) && !cudf::is_unsigned(type);
  };
  auto fn = [&]() -> trait_fn {
    switch (group_id) {
      case type_group_id::FLOATING_POINT: return cudf::is_floating_point;
      case type_group_id::INTEGRAL: return is_integral;
      case type_group_id::INTEGRAL_SIGNED: return is_integral_signed;
      case type_group_id::NUMERIC: return cudf::is_numeric;
      case type_group_id::TIMESTAMP: return cudf::is_timestamp;
      case type_group_id::DURATION: return cudf::is_duration;
      case type_group_id::FIXED_POINT: return cudf::is_fixed_point;
      case type_group_id::COMPOUND: return cudf::is_compound;
      case type_group_id::NESTED: return cudf::is_nested;
      default: CUDF_FAIL("Invalid data type group");
    }
  }();
  std::vector<cudf::type_id> types;
  for (int type_int = 0; type_int < static_cast<int32_t>(cudf::type_id::NUM_TYPE_IDS); ++type_int) {
    auto const type = static_cast<cudf::type_id>(type_int);
    if (type != cudf::type_id::EMPTY && fn(cudf::data_type(type))) types.push_back(type);
  }
  return types;
}

std::vector<cudf::type_id> get_type_or_group(std::vector<int32_t> const& ids)
{
  std::vector<cudf::type_id> all_type_ids;
  for (auto& id : ids) {
    auto const type_ids = get_type_or_group(id);
    all_type_ids.insert(std::end(all_type_ids), std::cbegin(type_ids), std::cend(type_ids));
  }
  return all_type_ids;
}
