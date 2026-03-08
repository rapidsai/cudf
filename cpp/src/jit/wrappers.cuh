
#pragma once
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/span>

namespace CUDF_EXPORT cudf {
namespace jit {

/**
 * @brief A column wrapper type that treats a column as a vector of elements.
 *
 */
template <typename Column>
struct vector_column_device_view : private Column {
  using base = Column;

  CUDF_HOST_DEVICE constexpr vector_column_device_view(Column const& src) : base{src} {}
  ~vector_column_device_view()                                           = default;
  vector_column_device_view(vector_column_device_view const&)            = default;
  vector_column_device_view(vector_column_device_view&&)                 = default;
  vector_column_device_view& operator=(vector_column_device_view const&) = default;
  vector_column_device_view& operator=(vector_column_device_view&&)      = default;

  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename T>
  CUDF_HOST_DEVICE T const* data() const noexcept
    requires(!base::is_mut)
  {
    return static_cast<T const*>(_data) + _offset;
  }

  template <typename T>
  CUDF_HOST_DEVICE T* data() const noexcept
    requires(base::is_mut)
  {
    return static_cast<T*>(_data) + _offset;
  }

  using base::is_null;
  using base::is_valid;
  using base::null_mask;

  template <typename T>
  [[nodiscard]] __device__ decltype(auto) element(size_type element) const noexcept
  {
    return data<T>()[element];
  }

  template <typename T>
  [[nodiscard]] __device__ cuda::std::optional<T> nullable_element(size_type element) const noexcept
  {
    if (is_null(element)) { return cuda::std::nullopt; }
    return element<T>(element);
  }

};

/**
 * @brief A column wrapper type that treats a column as a column of mutable strings.
 * The offsets will have been pre-initialized and the chars will have been pre-allocated.
 */
template <typename Column>
struct mut_strings_column_device_view : private Column {
  using base = Column;

  CUDF_HOST_DEVICE constexpr mut_strings_column_device_view(Column const& src) : base{src} {}

  ~mut_strings_column_device_view()                                               = default;
  mut_strings_column_device_view(mut_strings_column_device_view const&)            = default;
  mut_strings_column_device_view(mut_strings_column_device_view&&)                 = default;
  mut_strings_column_device_view& operator=(mut_strings_column_device_view const&) = default;
  mut_strings_column_device_view& operator=(mut_strings_column_device_view&&)      = default;

  using base::is_null;
  using base::is_valid;
  using base::null_mask;
  using base::nullable;
  using base::offset;
  using base::size;
  using base::type;

  template <typename T = cuda::std::span<char>>
  [[nodiscard]] __device__ cuda::std::span<char> element(size_type element) const noexcept
    requires(base::is_mut && cuda::std::is_same_v<T, cuda::std::span<char>>)
  {
    auto index   = element + offset();
    auto chars   = static_cast<char*>(_data);
    auto offsets = child(offsets_column_index);
    auto itr     = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    auto offset  = itr[index];
    return cuda::std::span<char>{chars + offset,
                                 static_cast<cudf::size_type>(itr[index + 1] - offset)};
  }

  template <typename T = cuda::std::span<char>>
  [[nodiscard]] __device__ cuda::std::optional<cuda::std::span<char>> nullable_element(
    size_type element) const noexcept
    requires(base::is_mut && cuda::std::is_same_v<T, cuda::std::span<char>>)
  {
    if (is_null(element)) { return cuda::std::nullopt; }
    return element(element);
  }

};

}  // namespace jit
}  // namespace CUDF_EXPORT cudf
