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
#include "random_distribution_factory.hpp"

#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform_scan.h>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <random>
#include <thread>
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
 * The implementation is SFINAEd for different type groups. Currently only used for fixed-width
 * types.
 */
template <typename T, typename Enable = void>
struct random_value_fn;

/**
 * @brief Creates an random timestamp/duration value
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<cudf::is_chrono<T>()>> {
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
      seconds_gen = [=](thrust::minstd_rand&, size_t size) {
        rmm::device_uvector<int64_t> result(size, rmm::cuda_stream_default);
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
    auto sec = seconds_gen(engine, size);
    auto ns  = nanoseconds_gen(engine, size);
    rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
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
 * @brief Creates an random fixed_point value. Not implemented yet.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<cudf::is_fixed_point<T>()>> {
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
    if (not scale.has_value()) {
      int const max_scale = std::numeric_limits<rep>::digits10;
      std::binomial_distribution<int> scale_dist_normal(2 * max_scale, 0.5);
      std::mt19937 engine;
      scale = numeric::scale_type{scale_dist_normal(engine) - max_scale};
    }
  }

  rmm::device_uvector<T> operator()(thrust::minstd_rand& engine, unsigned size)
  {
    auto ints = dist(engine, size);
    rmm::device_uvector<T> result(size, rmm::cuda_stream_default);
    // Clamp the generated random value to the specified range
    thrust::transform(thrust::device,
                      ints.begin(),
                      ints.end(),
                      result.begin(),
                      [scale       = *(this->scale),
                       upper_bound = this->upper_bound,
                       lower_bound = this->lower_bound] __device__(auto int_value) {
                        return T{std::max(std::min(int_value, upper_bound), lower_bound), scale};
                      });
    return result;
  }
};

/**
 * @brief Creates an random numeric value with the given distribution.
 */
template <typename T>
struct random_value_fn<
  T,
  typename std::enable_if_t<!std::is_same_v<T, bool> && cudf::is_numeric<T>()>> {
  T const lower_bound;
  T const upper_bound;
  distribution_fn<T> dist;

  random_value_fn(distribution_params<T> const& desc)
    : lower_bound{desc.lower_bound},
      upper_bound{desc.upper_bound},
      dist{make_distribution<T>(desc.id, desc.lower_bound, desc.upper_bound)}
  {
  }

  auto operator()(thrust::minstd_rand& engine, unsigned size)
  {
    // Clamp the generated random value to the specified range
    // return std::max(std::min(dist(engine), upper_bound), lower_bound);
    return dist(engine, size);
  }
};

struct valid_generator {
  thrust::minstd_rand engine;
  thrust::uniform_real_distribution<float> dist;
  float valid_prob;
  valid_generator(thrust::minstd_rand engine, float valid_probability)
    : engine(engine), dist{0, 1}, valid_prob{valid_probability}
  {
  }
  __device__ bool operator()(size_t n)
  {
    engine.discard(n);
    return dist(engine) < valid_prob;
  }
};

/**
 * @brief Creates an boolean value with given probability of returning `true`.
 */
template <typename T>
struct random_value_fn<T, typename std::enable_if_t<std::is_same_v<T, bool>>> {
  // bernoulli_distribution
  distribution_fn<bool> dist;

  random_value_fn(distribution_params<bool> const& desc)
    : dist{
        [valid_prob = desc.probability_true, dist = thrust::uniform_real_distribution<float>(0, 1)](
          thrust::minstd_rand& engine, size_t size) mutable -> rmm::device_uvector<bool> {
          rmm::device_uvector<bool> result(size, rmm::cuda_stream_default);
          thrust::tabulate(
            thrust::device, result.begin(), result.end(), valid_generator(engine, valid_prob));
          return result;
        }}
  {
  }
  auto operator()(thrust::minstd_rand& engine, unsigned size) { return dist(engine, size); }
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
// TODO Get rid of stored_as
template <typename T>
struct stored_as<T, typename std::enable_if_t<std::is_same_v<T, bool>>> {
  using type = bool;
};

template <typename T>
void printd(rmm::device_uvector<T>& dv)
{
  if constexpr (cudf::is_numeric<T>()) {
    thrust::host_vector<T> vec(thrust::device_pointer_cast(dv.begin()),
                               thrust::device_pointer_cast(dv.end()));
    std::cout << "[" << dv.size() << "] ";
    // for(auto& n : vec) { std::cout << n << " "; }
    thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(std::cout, " "));
    std::cout << std::endl;
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
  if constexpr (cudf::is_numeric<T>() or cudf::is_chrono<T>() or cudf::is_fixed_point<T>()) {
    // Working around vector<bool> and storing bools as int8_t
    using stored_Type = typename stored_as<T>::type;
    // bernoulli_distribution
    // get_null_frequency < 0 no null mask, ==0 all valids. //TODO remove unused computations
    auto valid_dist =
      random_value_fn<bool>(distribution_params<bool>{1. - profile.get_null_frequency()});
    auto value_dist = random_value_fn<T>{profile.get_distribution_params<T>()};

    auto const cardinality                      = std::min(num_rows, profile.get_cardinality());
    rmm::device_uvector<bool> samples_null_mask = valid_dist(engine, cardinality);
    rmm::device_uvector<stored_Type> samples    = value_dist(engine, cardinality);
    // std::cout << "cardinality: " << cardinality << std::endl;
    // std::cout << "num_rows: " << num_rows << std::endl;
    // std::cout << "samples: ";
    // printd(samples);

    // Distribution for picking elements from the array of samples
    std::uniform_int_distribution<cudf::size_type> sample_dist{0, cardinality - 1};
    auto const avg_run_len = profile.get_avg_run_length();
    // std::cout << "avg_run_len: " << avg_run_len << std::endl;
    // auto run_len_dist      = create_run_length_dist(avg_run_len);
    rmm::device_uvector<stored_Type> data(0, rmm::cuda_stream_default);
    rmm::device_uvector<bool> null_mask(0, rmm::cuda_stream_default);
    std::cout << "line " << __LINE__ << "\n";

    // Create a run-length values. (sum upto  num_rows) size~=num_rows/avg_run_len. could be more.
    // exclusive scan it. 1+run_len (account for data to be copied.)
    // generate indices(uniform), gather from cardinality vector.
    // transform_if lower_bound on run_len and copy that data, except for first.
    // TODO: Alternate ways/algo to generate this data???

    // generate size~=num_rows/run_len numbers gamma(). exclusive scan it. = ex_runlen.
    // generate uniform(0, cardinality-1) indices of size~=num_rows/run_len. - samples.
    // transform_if lower_bound on exrun_len, find index, collect from
    // cardinality_array[samples[index]];
    std::cout << "line " << __LINE__ << "\n";
    if (cardinality == 0) {
      data      = value_dist(engine, num_rows);
      null_mask = valid_dist(engine, num_rows);
    } else {
      auto sample_dist = random_value_fn<cudf::size_type>{
        distribution_params<cudf::size_type>{distribution_id::UNIFORM, 0, cardinality - 1}};
      auto avglen_dist = random_value_fn<int>{
        distribution_params<int>{distribution_id::UNIFORM, 1, 2 * avg_run_len}};
      std::cout << "line " << __LINE__ << "\n";
      if (avg_run_len > 1) {
        auto approx_run_len = num_rows / avg_run_len + 1;
        // std::cout << "approx_run_len: " << approx_run_len << std::endl;
        auto run_lens = avglen_dist(engine, approx_run_len);
        // std::cout << "run_lens: ";
        // printd(run_lens);
        thrust::inclusive_scan(
          thrust::device, run_lens.begin(), run_lens.end(), run_lens.begin(), std::plus<int>{});
        auto samples_indices = sample_dist(engine, approx_run_len + 1);
        // printd(run_lens);
        // std::cout << "samples_indices: ";
        // printd(samples_indices);

        data      = rmm::device_uvector<stored_Type>(num_rows, rmm::cuda_stream_default);
        null_mask = rmm::device_uvector<bool>(num_rows, rmm::cuda_stream_default);
        // This is gather.
        auto avg_repeated_sample_indices = thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          [rb              = run_lens.begin(),
           re              = run_lens.end(),
           samples_indices = samples_indices.begin()] __device__(cudf::size_type i) {
            auto sample_idx = thrust::upper_bound(thrust::seq, rb, re, i) - rb;
            return samples_indices[sample_idx];
          });
        std::cout << "line " << __LINE__ << "\n";
        thrust::gather(thrust::device,
                       avg_repeated_sample_indices,
                       avg_repeated_sample_indices + num_rows,
                       samples.begin(),
                       data.begin());
        std::cout << "line " << __LINE__ << "\n";
        thrust::gather(thrust::device,
                       avg_repeated_sample_indices,
                       avg_repeated_sample_indices + num_rows,
                       samples_null_mask.begin(),
                       null_mask.begin());
        std::cout << "line " << __LINE__ << "\n";
      } else {
        std::cout << "line " << __LINE__ << "\n";
        // generate n samples. and gather.
        auto samples_indices = sample_dist(engine, num_rows);
        data                 = rmm::device_uvector<stored_Type>(num_rows, rmm::cuda_stream_default);
        null_mask            = rmm::device_uvector<bool>(num_rows, rmm::cuda_stream_default);
        std::cout << "line " << __LINE__ << "\n";
        thrust::gather(thrust::device,
                       samples_indices.begin(),
                       samples_indices.end(),
                       samples.begin(),
                       data.begin());
        std::cout << "line " << __LINE__ << "\n";
        thrust::gather(thrust::device,
                       samples_indices.begin(),
                       samples_indices.end(),
                       samples_null_mask.begin(),
                       null_mask.begin());
      }
    }
    std::cout << "line " << __LINE__ << "\n";

    auto [result_bitmask, null_count] =
      cudf::detail::valid_if(null_mask.begin(), null_mask.end(), thrust::identity<bool>{});

    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_to_id<T>()},
      num_rows,
      data.release(),
      profile.get_null_frequency() < 0 ? rmm::device_buffer{} : std::move(result_bitmask));
  } else {
    std::cout << static_cast<int>(cudf::type_to_id<T>()) << std::endl;
    CUDF_FAIL("unsupported column type da");
  }
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

std::unique_ptr<cudf::column> create_random_utf8_string_column2(data_profile const& profile,
                                                                thrust::minstd_rand& engine,
                                                                cudf::size_type num_rows)
{
  auto len_dist =
    random_value_fn<uint32_t>{profile.get_distribution_params<cudf::string_view>().length_params};
  // TODO get_null_frequency == 0.
  auto valid_dist =
    random_value_fn<bool>(distribution_params<bool>{1. - profile.get_null_frequency()});
  auto lengths   = len_dist(engine, num_rows + 1);
  auto null_mask = valid_dist(engine, num_rows + 1);  // or cudf::bitmask_type
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
  rmm::device_uvector<cudf::size_type> offsets(num_rows + 1, rmm::cuda_stream_default);
  thrust::exclusive_scan(
    thrust::device, valid_lengths, valid_lengths + lengths.size(), offsets.begin());
  // offfsets are ready.
  auto chars_length = *thrust::device_pointer_cast(offsets.end() - 1);
  rmm::device_uvector<char> chars(chars_length, rmm::cuda_stream_default);
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
    profile.get_null_frequency() < 0 ? rmm::device_buffer{} : std::move(result_bitmask));
}

/**
 * @brief Create a UTF-8 string column with the average length.
 *
 */
/*
std::unique_ptr<cudf::column> create_random_utf8_string_column(data_profile const& profile,
 thrust::minstd_rand& engine,
 cudf::size_type num_rows)
{
 std::cout << "line " << __LINE__ << "\n";
 auto len_dist   =
random_value_fn<uint32_t>{profile.get_distribution_params<cudf::string_view>().length_params};
 std::cout<<"line "<<__LINE__<<"\n";
 auto valid_dist = random_value_fn<bool>(distribution_params<bool>{1. -
profile.get_null_frequency()}); std::cout<<"line "<<__LINE__<<"\n";

 auto lengths       = len_dist(engine, num_rows + 1);
 auto null_mask     = valid_dist(engine, num_rows + 1);  // or cudf::bitmask_type
 thrust::transform_if(thrust::device,
                     lengths.begin(),
                     lengths.end(),
                     null_mask.begin(),
                     lengths.begin(),
                     []__device__(auto) { return 0; },
                     thrust::logical_not<bool>{});
 auto valid_lengths = thrust::make_transform_iterator(
   thrust::make_zip_iterator(thrust::make_tuple(lengths.begin(), null_mask.begin())),
   valid_or_zero{});
 rmm::device_uvector<cudf::size_type> offsets(num_rows+1, rmm::cuda_stream_default);
 std::cout<<"line "<<__LINE__<<"\n";
 thrust::exclusive_scan(thrust::device, valid_lengths, valid_lengths+lengths.size(),
offsets.begin()); std::cout<<"line "<<__LINE__<<"\n"; auto chars_length =
*thrust::device_pointer_cast(offsets.end() - 1); auto char_dist    = random_value_fn<unsigned
char>{distribution_params<unsigned char>{distribution_id::UNIFORM, 32, 137}}; std::cout<<"line
[char] "<<__LINE__<<"\n"; auto chars = char_dist(engine, chars_length);  // simply generate 32-137.
// update prev char of any >=7F
 // fix UTF-8 codes.
 std::cout<<"line [after char]"<<__LINE__<<"\n";
 thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), chars.size()/2+1,
   [chars = chars.begin()] __device__ (int i) {
     if(chars[2*i] >= 0x7F) { // first byte is >= 0x7F
       chars[2*i+1] = chars[2*i]+1;
       chars[2*i]   = 0xC4;
     } else if(chars[2*i+1] >= 0x7F) { // second byte is >= 0x7F
       chars[2*i]   = 0xC4;
       chars[2*i+1] +=1;
     }
   });
 // offset boundary fix.
 std::cout<<"line "<<__LINE__<<"\n";
 thrust::for_each(thrust::device, offsets.begin(), offsets.end()-1,
  [chars = chars.begin()] __device__ (cudf::size_type offset) {
   if(offset>0) {
     if(chars[offset-1]==0xC4) {
       chars[offset-1] =  ' ';
       chars[offset  ] =  ' ';
     }
   }
 });
 std::cout<<"line "<<__LINE__<<"\n";
 auto [result_bitmask, null_count] = cudf::detail::valid_if(null_mask.begin(), null_mask.end()-1,
thrust::identity<bool>{}); rmm::device_uvector<char> uchars(chars_length, rmm::cuda_stream_default);
 thrust::copy(thrust::device, chars.begin(), chars.end(), uchars.begin());
 return cudf::make_strings_column(num_rows, std::move(offsets), std::move(uchars),
std::move(result_bitmask));
}
*/

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
  // ~10% UTF-8. ~90% ASCII.
  // ~20% space, ~80% space.
  // range 32-127 is ASCII; 127-136 will be multi-byte UTF-8
  auto const cardinality = std::min(profile.get_cardinality(), num_rows);
  auto const avg_run_len = profile.get_avg_run_length();

  // TODO check if null is working fine.
  auto sample_strings =
    create_random_utf8_string_column2(profile, engine, cardinality == 0 ? num_rows : cardinality);
  if (cardinality == 0) {
    return sample_strings;
  } else {
    auto sample_dist = random_value_fn<cudf::size_type>{
      distribution_params<cudf::size_type>{distribution_id::UNIFORM, 0, cardinality - 1}};
    std::cout << "line " << __LINE__ << "\n";
    auto sample_indices = [&]() {
      if (avg_run_len > 1) {
        std::cout << "line " << __LINE__ << "\n";
        auto avglen_dist = random_value_fn<int>{
          distribution_params<int>{distribution_id::UNIFORM, 1, 2 * avg_run_len}};
        auto approx_run_len = num_rows / avg_run_len + 1;
        auto run_lens       = avglen_dist(engine, approx_run_len);
        std::cout << "line " << __LINE__ << "\n";
        thrust::inclusive_scan(
          thrust::device, run_lens.begin(), run_lens.end(), run_lens.begin(), std::plus<int>{});
        std::cout << "line " << __LINE__ << "\n";
        auto samples_indices = sample_dist(engine, approx_run_len + 1);
        std::cout << "line " << __LINE__ << "\n";
        auto avg_repeated_sample_indices = thrust::make_transform_iterator(
          thrust::make_counting_iterator(0),
          [rb              = run_lens.begin(),
           re              = run_lens.end(),
           samples_indices = samples_indices.begin()] __device__(cudf::size_type i) {
            auto sample_idx = thrust::upper_bound(thrust::seq, rb, re, i) - rb;
            return samples_indices[sample_idx];
          });
        rmm::device_uvector<cudf::size_type> repeated_sample_indices(num_rows,
                                                                     rmm::cuda_stream_default);
        std::cout << "line " << __LINE__ << "\n";
        thrust::copy(thrust::device,
                     avg_repeated_sample_indices,
                     avg_repeated_sample_indices + num_rows,
                     repeated_sample_indices.begin());
        std::cout << "line " << __LINE__ << "\n";
        return repeated_sample_indices;
      } else {
        std::cout << "line " << __LINE__ << "\n";
        auto samples_indices = sample_dist(engine, num_rows);
        std::cout << "line " << __LINE__ << "\n";
        return samples_indices;
      }
    }();
    std::cout << "line " << __LINE__ << "\n";
    auto [free_b, total_b] =
      rmm::mr::get_current_device_resource()->get_mem_info(rmm::cuda_stream_default);
    std::cout << "free= " << free_b << " used= " << total_b - free_b << " total= " << total_b
              << "\n";
    std::cout << "sample_indices.size():" << sample_indices.size() << "\n";
    std::cout << "cardinality: " << cardinality << "\n";
    // thrust::host_vector<cudf::size_type> h(thrust::device_pointer_cast(sample_indices.begin()),
    // thrust::device_pointer_cast(sample_indices.end())); thrust::copy(h.begin(), h.end(),
    // std::ostream_iterator<cudf::size_type>(std::cout, "\t"));
    std::cout << "sample_strings->size(): " << sample_strings->size() << "\n";
    auto scv = cudf::strings_column_view{sample_strings->view()};
    std::cout << "scv: " << scv.chars().size() << " " << scv.offsets().size() << "\n";
    // cudf::test::print(sample_strings->view());
    // cudf::test::print(cudf::bit_cast(scv.chars(), cudf::data_type{cudf::type_id::UINT8}));
    // cudf::test::print(scv.chars());
    // cudf::test::print(scv.offsets());
    // gather
    auto str_table =
      cudf::detail::gather(cudf::table_view{{sample_strings->view()}},
                           sample_indices,
                           cudf::out_of_bounds_policy::DONT_CHECK,  // TODO ensure no memory errors.
                           cudf::detail::negative_index_policy::NOT_ALLOWED);
    std::cout << "line " << __LINE__ << "\n";
    return std::move(str_table->release()[0]);
  }
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::dictionary32>(data_profile const& profile,
                                                                       thrust::minstd_rand& engine,
                                                                       cudf::size_type num_rows)
{
  CUDF_FAIL("not implemented yet");
}

template <>
std::unique_ptr<cudf::column> create_random_column<cudf::struct_view>(data_profile const& profile,
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

template <typename T>
struct clamp_down : public thrust::unary_function<T, T> {
  T max_size;
  clamp_down(T max_size) : max_size(max_size) {}
  __host__ __device__ T operator()(T x) const { return min(x, max_size); }
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
  auto valid_dist =
    random_value_fn<bool>(distribution_params<bool>{1. - profile.get_null_frequency()});
  std::cout << "line " << __LINE__ << "\n";

  // Generate the list column bottom-up
  auto list_column = std::move(leaf_column);
  for (int lvl = 0; lvl < dist_params.max_depth; ++lvl) {
    // Generating the next level - offsets point into the current list column
    auto current_child_column      = std::move(list_column);
    cudf::size_type const num_rows = current_child_column->size() / single_level_mean;
    std::cout << "num_rows: " << num_rows << "\n";

    std::cout << "line " << __LINE__ << "\n";
    auto offsets = len_dist(engine, num_rows + 1);
    auto valids  = valid_dist(engine, num_rows);
    std::cout << "line " << __LINE__ << "\n";
    std::cout << "offsets.size(): " << offsets.size() << "\n";
    // to ensure these values <= current_child_column->size()
    auto output_offsets = thrust::make_transform_output_iterator(
      offsets.begin(), clamp_down{current_child_column->size()});

    thrust::exclusive_scan(thrust::device, offsets.begin(), offsets.end(), output_offsets);
    thrust::device_pointer_cast(offsets.end())[-1] =
      current_child_column->size();  // Always include all elements

    std::cout << "line " << __LINE__ << "\n";
    auto offsets_column = std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::INT32}, num_rows + 1, offsets.release());
    std::cout << "line " << __LINE__ << "\n";
    std::cout << "offsets_column: " << offsets_column->size() << "\n";
    // cudf::test::print(*offsets_column);
    std::cout << "line " << __LINE__ << "\n";

    auto [null_mask, null_count] =
      cudf::detail::valid_if(valids.begin(), valids.end(), thrust::identity<bool>{});
    list_column = cudf::make_lists_column(
      num_rows,
      std::move(offsets_column),
      std::move(current_child_column),
      profile.get_null_frequency() < 0 ? 0 : null_count,  // cudf::UNKNOWN_NULL_COUNT,
      profile.get_null_frequency() < 0 ? rmm::device_buffer{} : std::move(null_mask));
    std::cout << "line " << __LINE__ << "\n";
  }
  std::cout << "line " << __LINE__ << "\n";
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

  // auto const processor_count            = std::thread::hardware_concurrency();
  // cudf::size_type const cols_per_thread = (num_cols + processor_count - 1) / processor_count;
  // cudf::size_type next_col              = 0;
  // std::vector<std::future<columns_vector>> col_futures;
  // random_value_fn<unsigned> seed_dist(
  //   {distribution_id::UNIFORM, 0, std::numeric_limits<unsigned>::max()});
  thrust::uniform_int_distribution<unsigned> seed_dist;
  // for (unsigned int i = 0; i < processor_count && next_col < num_cols; ++i) {
  //   auto thread_engine         = deterministic_engine(seed_dist(seed_engine));
  //   auto const thread_num_cols = std::min(num_cols - next_col, cols_per_thread);
  //   std::vector<cudf::type_id> thread_types(out_dtype_ids.begin() + next_col,
  //                                           out_dtype_ids.begin() + next_col + thread_num_cols);
  //   col_futures.emplace_back(std::async(std::launch::async,
  //                                       create_random_columns,
  //                                       std::cref(profile),
  //                                       std::move(thread_types),
  //                                       std::move(thread_engine),
  //                                       num_rows.count));
  //   next_col += thread_num_cols;
  // }

  // columns_vector output_columns;
  // for (auto& cf : col_futures) {
  //   auto partial_table = cf.get();
  //   output_columns.reserve(output_columns.size() + partial_table.size());
  //   std::move(
  //     std::begin(partial_table), std::end(partial_table), std::back_inserter(output_columns));
  //   partial_table.clear();
  // }
  columns_vector output_columns;
  std::transform(out_dtype_ids.begin(),
                 out_dtype_ids.end(),
                 std::back_inserter(output_columns),
                 [&](auto tid) mutable {
                   auto engine = deterministic_engine(seed_dist(seed_engine));
                   return cudf::type_dispatcher(
                     cudf::data_type(tid), create_rand_col_fn{}, profile, engine, num_rows.count);
                 });
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
