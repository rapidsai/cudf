
#include "types.hpp"

namespace rmm {
class device_buffer;
}

namespace cudf {

class bitmask_view {
 public:
  __device__ bool is_valid(size_type i) const noexcept {
    // FIXME Implement
    return true;
  }

  __device__ bool is_null(size_type i) const noexcept {
    return not is_valid(i);
  }

  __host__ __device__ bool nullable() const noexcept {
    return nullptr != _mask;
  }

  __host__ __device__ bitmask_type* data() noexcept { return _mask; }
  __host__ __device__ bitmask_type const* data() const noexcept {
    return _mask;
  }

 private:
  bitmask_type* _mask{nullptr};
  size_type _length{0};
};

class bitmask {
 public:
 private:
  rmm::device_buffer data{};
  size_type length{};
};

}  // namespace cudf