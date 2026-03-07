
#pragma once
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>
#include <cuda/std/span>

namespace CUDF_EXPORT cudf {
namespace jit {

/**
 * @brief A column wrapper type that treats a column as a span of elements. This treats the element
 * as a contiguous sequence of elements.
 *
 */
template <typename Column>
struct span_column : private Column {
  using base = Column;

  CUDF_HOST_DEVICE constexpr span_column(Column const& src) : base{src} {}
  ~span_column()                             = default;
  span_column(span_column const&)            = default;
  span_column(span_column&&)                 = default;
  span_column& operator=(span_column const&) = default;
  span_column& operator=(span_column&&)      = default;

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

  CUDF_HOST_DEVICE Column& to_base() { return static_cast<Column&>(*this); }

  CUDF_HOST_DEVICE Column const& to_base() const { return static_cast<Column const&>(*this); }
};

/**
 * @brief A column wrapper type that treats a column as a column of mutable strings.
 *
 */
template <typename Column>
struct mut_string_column : private Column {
  using base = Column;

  CUDF_HOST_DEVICE constexpr mut_string_column(Column const& src) : base{src} {}

  ~mut_string_column()                                   = default;
  mut_string_column(mut_string_column const&)            = default;
  mut_string_column(mut_string_column&&)                 = default;
  mut_string_column& operator=(mut_string_column const&) = default;
  mut_string_column& operator=(mut_string_column&&)      = default;

  using base::is_null;
  using base::is_valid;
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

  CUDF_HOST_DEVICE Column& to_base() { return static_cast<Column&>(*this); }

  CUDF_HOST_DEVICE Column const& to_base() const { return static_cast<Column const&>(*this); }
};

}  // namespace jit
}  // namespace CUDF_EXPORT cudf
