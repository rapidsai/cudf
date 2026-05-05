
#include <cudf/strings/string_view.cuh>

#include <cuda/std/optional>

#include <cstdint>

__device__ inline bool strcontains(cudf::string_view text, cudf::string_view token)
{
  return text.find(token) != cudf::string_view::npos;
}

template <typename T>
__device__ inline T load(void const* inputs, int input_stride, int arg)
{
  auto p = reinterpret_cast<T const*>(static_cast<char const*>(inputs) + arg * input_stride);
  return *p;
}

template <typename T>
__device__ inline void store(void* outputs, int output_stride, int arg, T value)
{
  auto p = reinterpret_cast<T*>(static_cast<char*>(outputs) + arg * output_stride);
  *p     = value;
}

extern "C" __device__ int operation(
  void*, long int, void const* inputs, int input_stride, void* outputs, int output_stride)
{
  // Input schema:
  //   0: price(double),
  //   1: qty(int32),
  //   2: discount(float),
  //   3: tax_rate(float),
  //   4: ship_mode(string_view),
  //   5: comment(string_view)
  auto price    = load<double>(inputs, input_stride, 0);
  auto qty      = load<int32_t>(inputs, input_stride, 1);
  auto discount = load<float>(inputs, input_stride, 2);
  auto tax_rate = load<float>(inputs, input_stride, 3);
  auto ship     = load<cudf::string_view>(inputs, input_stride, 4);
  auto comment  = load<cudf::string_view>(inputs, input_stride, 5);

  // Parameters:
  //  6: priority_threshold(double),
  //  7: require_expedited(bool),
  //  8: promo_discount_threshold(float)
  auto priority_threshold       = load<double>(inputs, input_stride, 6);
  auto require_expedited        = load<bool>(inputs, input_stride, 7);
  auto promo_discount_threshold = load<float>(inputs, input_stride, 8);

  auto net   = price * qty * (1.0 - discount);
  auto gross = net * (1.0 + tax_rate);

  auto expedited = strcontains(ship, cudf::string_view{"expedited", 9});
  auto priority  = (gross > priority_threshold) && (!require_expedited || expedited);

  auto has_promo_keyword = strcontains(comment, cudf::string_view{"promo", 5}) ||
                           strcontains(comment, cudf::string_view{"sale", 4}) ||
                           strcontains(comment, cudf::string_view{"coupon", 6});

  auto is_promo = has_promo_keyword || (discount >= promo_discount_threshold);

  store(outputs, output_stride, 0, net);
  store(outputs, output_stride, 1, gross);
  store(outputs, output_stride, 2, priority);
  store(outputs, output_stride, 3, is_promo);

  return 0;
}
