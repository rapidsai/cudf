#include "types.hpp"

namespace cudf {
struct column_view {
  column_view(void* data, Dtype type, size_type length,
              bitmask_view const& mask, size_type null_count)
      : _data{data},
        _type{type},
        _length{length},
        _mask{mask},
        _null_count{null_count} {}

  template <typename T>
  __host__ __device__ T* typed_data() noexcept {
    return static_cast<T*>(data);
  }

  template <typename T>
  __host__ __device__ T const* typed_data() const noexcept {
    return static_cast<T const*>(data);
  }

  __host__ __device__ void const* data() noexcept { return _data; }
  __host__ __device__ void* data() const noexcept { return _data; }

  __device__ bool is_valid(size_type i) const noexcept {
    return _mask.is_valid(i);
  }

  __device__ bool is_null(size_type i) const noexcept {
    return _mask.is_null(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return _mask.nullable();
  }

  __host__ __device__ size_type null_count() const noexcept {
    return _null_count;
  }

  __host__ __device__ size_type length() const noexcept { return _length; }

  __host__ __device__ DType type() const noexcept { return _type; }

  __host__ __device__ bitmask_view mask() noexcept { return _mask; }
  __host__ __device__ bitmask_view const mask() const noexcept { return _mask; }

  __host__ __device__ column_view* other() noexcept { return _other; }
  __host__ __device__ column_view const* other() const noexcept {
    return _other;
  }

 private:
  void* _data{nullptr};
  DType _type{INVALID};
  cudf::size_type _length{0};
  bitmask_view _mask;
  cudf::size_type _null_count{0};
  column_view* _other{nullptr};
};
}  // namespace cudf