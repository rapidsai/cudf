/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "generate_benchmark_input.hpp"
#include "random_distribution_factory.hpp"

#include <cudf/column/column.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/bit.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <future>
#include <memory>
#include <random>
#include <thread>
#include <vector>

/**
 * @brief Mersenne Twister pseudo-random engine.
 */
auto deterministic_engine(unsigned seed) { return std::mt19937{seed}; }

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

// Utilities to determine the mean size of an element, given the data profile
template <typename T>
std::enable_if_t<cudf::is_fixed_width<T>(), size_t> avg_element_size(data_profile const& profile)
{
  return sizeof(T);
}

template <typename T>
std::enable_if_t<!cudf::is_fixed_width<T>(), size_t> avg_element_size(data_profile const& profile)
{
  CUDF_FAIL("not implemented!");
}

template <>
size_t avg_element_size<cudf::string_view>(data_profile const& profile)
{
  auto const dist = profile.get_distribution_params<cudf::string_view>().length_params;
  return get_distribution_mean(dist);
}

template <>
size_t avg_element_size<cudf::list_view>(data_profile const& profile)
{
  auto const dist_params       = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean = get_distribution_mean(dist_params.length_params);
  auto const element_size      = cudf::size_of(cudf::data_type{dist_params.element_type});
  return element_size * pow(single_level_mean, dist_params.max_depth);
}

struct avg_element_size_fn {
  template <typename T>
  size_t operator()(data_profile const& profile)
  {
    return avg_element_size<T>(profile);
  }
};

size_t avg_element_bytes(data_profile const& profile, cudf::type_id tid)
{
  return cudf::type_dispatcher(cudf::data_type(tid), avg_element_size_fn{}, profile);
}

/**
 * @brief Functor that computes a random column element with the given data profile.
 *
 * The implementation is SFINAEd for diffent type groups. Currently only used for fixed-width types.
 */
template <typename T, typename Enable = void>
struct random_value_fn;

/**
 * @brief Creates an random timestamp/duration value
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<cudf::is_chrono<T>()>> {
  std::function<int64_t(std::mt19937&)> seconds_gen;
  std::function<int64_t(std::mt19937&)> nanoseconds_gen;

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
      seconds_gen = [=](std::mt19937&) { return range_s.second.count(); };

      std::pair<cudf::duration_ns, cudf::duration_ns> const range_ns = {
        duration_cast<cudf::duration_ns>(typename T::duration{params.lower_bound}),
        duration_cast<cudf::duration_ns>(typename T::duration{params.upper_bound})};
      nanoseconds_gen = make_distribution<int64_t>(distribution_id::UNIFORM,
                                                   std::min(range_ns.first.count(), 0l),
                                                   std::max(range_ns.second.count(), 0l));
    }
  }

  T operator()(std::mt19937& engine)
  {
    auto const timestamp_ns =
      cudf::duration_s{seconds_gen(engine)} + cudf::duration_ns{nanoseconds_gen(engine)};
    // Return value in the type's precision
    return T(cuda::std::chrono::duration_cast<typename T::duration>(timestamp_ns));
  }
};

/**
 * @brief Creates an random fixed_point value. Not implemented yet.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<cudf::is_fixed_point<T>()>> {
  random_value_fn(distribution_params<T> const&) {}
  T operator()(std::mt19937& engine) { CUDF_FAIL("Not implemented"); }
};

/**
 * @brief Creates an random numeric value with the given distribution.
 */
template <typename T>
struct random_value_fn<
  T,
  typename std::enable_if_t<!std::is_same<T, bool>::value && cudf::is_numeric<T>()>> {
  T const lower_bound;
  T const upper_bound;
  distribution_fn<T> dist;

  random_value_fn(distribution_params<T> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<T>(desc.id, desc.lower_bound, desc.upper_bound)}
  {
  }

  T operator()(std::mt19937& engine)
  {
    // Clamp the generated random value to the specified range
    return std::max(std::min(dist(engine), upper_bound), lower_bound);
  }
};

/**
 * @brief Creates an boolean value with given probability of returning `true`.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<std::is_same<T, bool>::value>> {
  std::bernoulli_distribution b_dist;

  random_value_fn(distribution_params<bool> const& desc) : b_dist{desc.probability_true} {}
  bool operator()(std::mt19937& engine) { return b_dist(engine); }
};

size_t null_mask_size(cudf::size_type num_rows)
{
  constexpr size_t bitmask_bits = cudf::detail::size_in_bits<cudf::bitmask_type>();
  return (num_rows + bitmask_bits - 1) / bitmask_bits;
}
template <typename T>
void set_element_at(T value,
                    bool valid,
                    std::vector<T>& values,
                    std::vector<cudf::bitmask_type>& null_mask,
                    cudf::size_type idx)
{
  if (valid) {
    values[idx] = value;
  } else {
    cudf::clear_bit_unsafe(null_mask.data(), idx);
  }
}

auto create_run_length_dist(cudf::size_type avg_run_len)
{
  // Distribution with low probability of generating 0-1 even with a low `avg_run_len` value
  static constexpr float alpha = 4.f;
  return std::gamma_distribution<float>{alpha, avg_run_len / alpha};
}

// identity mapping, except for bools
template <typename T, typename Enable = void>
struct stored_as {
  using type = T;
};

// Use `int8_t` for bools because that's how they're stored in columns
template <typename T>
struct stored_as<T, typename std::enable_if_t<std::is_same<T, bool>::value>> {
  using type = int8_t;
};

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
                                                   std::mt19937& engine,
                                                   cudf::size_type num_rows)
{
  // Working around vector<bool> and storing bools as int8_t
  using stored_Type = typename stored_as<T>::type;

  auto valid_dist = std::bernoulli_distribution{1. - profile.get_null_frequency()};
  auto value_dist = random_value_fn<T>{profile.get_distribution_params<T>()};

  auto const cardinality = std::min(num_rows, profile.get_cardinality());
  std::vector<stored_Type> samples(cardinality);
  std::vector<cudf::bitmask_type> samples_null_mask(null_mask_size(cardinality), ~0);
  for (cudf::size_type si = 0; si < cardinality; ++si) {
    set_element_at(
      (stored_Type)value_dist(engine), valid_dist(engine), samples, samples_null_mask, si);
  }

  // Distribution for picking elements from the array of samples
  std::uniform_int_distribution<cudf::size_type> sample_dist{0, cardinality - 1};
  auto const avg_run_len = profile.get_avg_run_length();
  auto run_len_dist      = create_run_length_dist(avg_run_len);
  std::vector<stored_Type> data(num_rows);
  std::vector<cudf::bitmask_type> null_mask(null_mask_size(num_rows), ~0);

  for (cudf::size_type row = 0; row < num_rows; ++row) {
    if (cardinality == 0) {
      set_element_at((stored_Type)value_dist(engine), valid_dist(engine), data, null_mask, row);
    } else {
      auto const sample_idx = sample_dist(engine);
      set_element_at(samples[sample_idx],
                     cudf::bit_is_set(samples_null_mask.data(), sample_idx),
                     data,
                     null_mask,
                     row);
    }

    if (avg_run_len > 1) {
      int const run_len = std::min<int>(num_rows - row, std::round(run_len_dist(engine)));
      for (int offset = 1; offset < run_len; ++offset) {
        set_element_at(
          data[row], cudf::bit_is_set(null_mask.data(), row), data, null_mask, row + offset);
      }
      row += std::max(run_len - 1, 0);
    }
  }

  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_to_id<T>()},
    num_rows,
    rmm::device_buffer(data.data(), num_rows * sizeof(stored_Type), rmm::cuda_stream_default),
    rmm::device_buffer(
      null_mask.data(), null_mask.size() * sizeof(cudf::bitmask_type), rmm::cuda_stream_default));
}

/**
 * @brief Class that holds string column data in host memory.
 */
struct string_column_data {
  std::vector<char> chars;
  std::vector<cudf::size_type> offsets;
  std::vector<cudf::bitmask_type> null_mask;
  explicit string_column_data(cudf::size_type rows, cudf::size_type size)
  {
    offsets.reserve(rows + 1);
    offsets.push_back(0);
    chars.reserve(size);
    null_mask.insert(null_mask.end(), null_mask_size(rows), ~0);
  }
};

/**
 * @brief Copy a string from one host-side "column" to another.
 *
 * Assumes that the destination null mask is initialized with all bits valid.
 */
void copy_string(cudf::size_type src_idx,
                 string_column_data const& src,
                 cudf::size_type dst_idx,
                 string_column_data& dst)
{
  if (!cudf::bit_is_set(src.null_mask.data(), src_idx))
    cudf::clear_bit_unsafe(dst.null_mask.data(), dst_idx);
  auto const str_len = src.offsets[src_idx + 1] - src.offsets[src_idx];
  dst.chars.resize(dst.chars.size() + str_len);
  if (cudf::bit_is_set(src.null_mask.data(), src_idx)) {
    std::copy_n(
      src.chars.begin() + src.offsets[src_idx], str_len, dst.chars.begin() + dst.offsets.back());
  }
  dst.offsets.push_back(dst.chars.size());
}

/**
 * @brief Generate a random string at the end of the host-side "column".
 *
 * Assumes that the destination null mask is initialized with all bits valid.
 */
template <typename Char_gen>
void append_string(Char_gen& char_gen, bool valid, uint32_t length, string_column_data& column_data)
{
  if (!valid) {
    auto const idx = column_data.offsets.size() - 1;
    cudf::clear_bit_unsafe(column_data.null_mask.data(), idx);
    // duplicate the offset value to indicate an empty row
    column_data.offsets.push_back(column_data.offsets.back());
    return;
  }
  for (uint32_t idx = 0; idx < length; ++idx) {
    auto const ch = char_gen();
    if (ch >= '\x7F')                       // x7F is at the top edge of ASCII
      column_data.chars.push_back('\xC4');  // these characters are assigned two bytes
    column_data.chars.push_back(static_cast<char>(ch + (ch >= '\x7F')));
  }
  column_data.offsets.push_back(column_data.chars.size());
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
                                                                      std::mt19937& engine,
                                                                      cudf::size_type num_rows)
{
  auto char_dist = [&engine,  // range 32-127 is ASCII; 127-136 will be multi-byte UTF-8
                    dist = std::uniform_int_distribution<unsigned char>{32, 137}]() mutable {
    return dist(engine);
  };
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::string_view>().length_params};
  auto valid_dist = std::bernoulli_distribution{1. - profile.get_null_frequency()};

  auto const avg_string_len = avg_element_size<cudf::string_view>(profile);
  auto const cardinality    = std::min(profile.get_cardinality(), num_rows);
  string_column_data samples(cardinality, cardinality * avg_string_len);
  for (cudf::size_type si = 0; si < cardinality; ++si) {
    append_string(char_dist, valid_dist(engine), len_dist(engine), samples);
  }

  auto const avg_run_len = profile.get_avg_run_length();
  auto run_len_dist      = create_run_length_dist(avg_run_len);

  string_column_data out_col(num_rows, num_rows * avg_string_len);
  std::uniform_int_distribution<cudf::size_type> sample_dist{0, cardinality - 1};
  for (cudf::size_type row = 0; row < num_rows; ++row) {
    if (cardinality == 0) {
      append_string(char_dist, valid_dist(engine), len_dist(engine), out_col);
    } else {
      copy_string(sample_dist(engine), samples, row, out_col);
    }
    if (avg_run_len > 1) {
      int const run_len = std::min<int>(num_rows - row, std::round(run_len_dist(engine)));
      for (int offset = 1; offset < run_len; ++offset) {
        copy_string(row, out_col, row + offset, out_col);
      }
      row += std::max(run_len - 1, 0);
    }
  }

  auto d_chars     = cudf::detail::make_device_uvector_sync(out_col.chars);
  auto d_offsets   = cudf::detail::make_device_uvector_sync(out_col.offsets);
  auto d_null_mask = cudf::detail::make_device_uvector_sync(out_col.null_mask);
  return cudf::make_strings_column(d_chars, d_offsets, d_null_mask);
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::dictionary32>(data_profile const& profile,
                                                                       std::mt19937& engine,
                                                                       cudf::size_type num_rows)
{
  CUDF_FAIL("not implemented yet");
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::struct_view>(data_profile const& profile,
                                                                      std::mt19937& engine,
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
                                           std::mt19937& engine,
                                           cudf::size_type num_rows)
  {
    return create_random_column<T>(profile, engine, num_rows);
  }
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
                                                                    std::mt19937& engine,
                                                                    cudf::size_type num_rows)
{
  auto const dist_params       = profile.get_distribution_params<cudf::list_view>();
  auto const single_level_mean = get_distribution_mean(dist_params.length_params);
  auto const num_elements      = num_rows * pow(single_level_mean, dist_params.max_depth);

  auto leaf_column = cudf::type_dispatcher(
    cudf::data_type(dist_params.element_type), create_rand_col_fn{}, profile, engine, num_elements);
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::list_view>().length_params};
  auto valid_dist = std::bernoulli_distribution{1. - profile.get_null_frequency()};

  // Generate the list column bottom-up
  auto list_column = std::move(leaf_column);
  for (int lvl = 0; lvl < dist_params.max_depth; ++lvl) {
    // Generating the next level - offsets point into the current list column
    auto current_child_column      = std::move(list_column);
    cudf::size_type const num_rows = current_child_column->size() / single_level_mean;

    std::vector<int32_t> offsets{0};
    offsets.reserve(num_rows + 1);
    std::vector<cudf::bitmask_type> null_mask(null_mask_size(num_rows), ~0);
    for (cudf::size_type row = 1; row < num_rows + 1; ++row) {
      offsets.push_back(
        std::min<int32_t>(current_child_column->size(), offsets.back() + len_dist(engine)));
      if (!valid_dist(engine)) cudf::clear_bit_unsafe(null_mask.data(), row);
    }
    offsets.back() = current_child_column->size();  // Always include all elements

    auto offsets_column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32},
      offsets.size(),
      rmm::device_buffer(
        offsets.data(), offsets.size() * sizeof(int32_t), rmm::cuda_stream_default));

    list_column = cudf::make_lists_column(
      num_rows,
      std::move(offsets_column),
      std::move(current_child_column),
      cudf::UNKNOWN_NULL_COUNT,
      rmm::device_buffer(
        null_mask.data(), null_mask.size() * sizeof(cudf::bitmask_type), rmm::cuda_stream_default));
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
                                     std::mt19937 engine,
                                     cudf::size_type num_rows)
{
  columns_vector output_columns;
  std::transform(
    dtype_ids.begin(), dtype_ids.end(), std::back_inserter(output_columns), [&](auto tid) {
      return cudf::type_dispatcher(
        cudf::data_type(tid), create_rand_col_fn{}, profile, engine, num_rows);
    });
  return output_columns;
}

/**
 * @brief Repeats the input data types in round-robin order to fill a vector of @ref num_cols
 * elements.
 */
std::vector<cudf::type_id> repeat_dtypes(std::vector<cudf::type_id> const& dtype_ids,
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
                                                 cudf::size_type num_cols,
                                                 table_size_bytes table_bytes,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  auto const out_dtype_ids = repeat_dtypes(dtype_ids, num_cols);
  size_t const avg_row_bytes =
    std::accumulate(out_dtype_ids.begin(), out_dtype_ids.end(), 0ul, [&](size_t sum, auto tid) {
      return sum + avg_element_bytes(profile, tid);
    });
  cudf::size_type const num_rows = table_bytes.size / avg_row_bytes;

  return create_random_table(out_dtype_ids, num_cols, row_count{num_rows}, profile, seed);
}

std::unique_ptr<cudf::table> create_random_table(std::vector<cudf::type_id> const& dtype_ids,
                                                 cudf::size_type num_cols,
                                                 row_count num_rows,
                                                 data_profile const& profile,
                                                 unsigned seed)
{
  auto const out_dtype_ids = repeat_dtypes(dtype_ids, num_cols);
  auto seed_engine         = deterministic_engine(seed);

  auto const processor_count            = std::thread::hardware_concurrency();
  cudf::size_type const cols_per_thread = (num_cols + processor_count - 1) / processor_count;
  cudf::size_type next_col              = 0;
  std::vector<std::future<columns_vector>> col_futures;
  random_value_fn<unsigned> seed_dist(
    {distribution_id::UNIFORM, 0, std::numeric_limits<unsigned>::max()});
  for (unsigned int i = 0; i < processor_count && next_col < num_cols; ++i) {
    auto thread_engine         = deterministic_engine(seed_dist(seed_engine));
    auto const thread_num_cols = std::min(num_cols - next_col, cols_per_thread);
    std::vector<cudf::type_id> thread_types(out_dtype_ids.begin() + next_col,
                                            out_dtype_ids.begin() + next_col + thread_num_cols);
    col_futures.emplace_back(std::async(std::launch::async,
                                        create_random_columns,
                                        std::cref(profile),
                                        std::move(thread_types),
                                        std::move(thread_engine),
                                        num_rows.count));
    next_col += thread_num_cols;
  }

  columns_vector output_columns;
  for (auto& cf : col_futures) {
    auto partial_table = cf.get();
    output_columns.reserve(output_columns.size() + partial_table.size());
    std::move(
      std::begin(partial_table), std::end(partial_table), std::back_inserter(output_columns));
    partial_table.clear();
  }

  return std::make_unique<cudf::table>(std::move(output_columns));
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
