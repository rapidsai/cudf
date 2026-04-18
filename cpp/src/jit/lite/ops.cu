#include <jit/lite/lite.cuh>

namespace __attribute__((visibility("default"))) cudf
{
  namespace lite {

  // TODO: pre-instantiate some of the operators so they can be used in LTO

#define DEFINE_UNARY_OP(NAME, RET_TYPE, TYPE, TYPE_TAG)                             \
  extern "C" __device__ int cudf_##NAME##__##TYPE_TAG(RET_TYPE* out, TYPE const* a) \
  {                                                                                 \
    return operators::NAME<TYPE>(out, a);                                           \
  }

#define DEFINE_BINARY_OP(NAME, RET_TYPE, TYPE, TYPE_TAG)                                           \
  extern "C" __device__ int cudf_##NAME##__##TYPE_TAG(RET_TYPE* out, TYPE const* a, TYPE const* b) \
  {                                                                                                \
    return operators::NAME<TYPE>(out, a, b);                                                       \
  }

  DEFINE_UNARY_OP(abs, int8_t, int8_t, i8)
  DEFINE_UNARY_OP(abs, optional<int8_t>, optional<int8_t>, i8_opt)
  DEFINE_UNARY_OP(abs, int16_t, int16_t, i16)
  DEFINE_UNARY_OP(abs, optional<int16_t>, optional<int16_t>, i16_opt)
  DEFINE_UNARY_OP(abs, int32_t, int32_t, i32)
  DEFINE_UNARY_OP(abs, optional<int32_t>, optional<int32_t>, i32_opt)
  DEFINE_UNARY_OP(abs, int64_t, int64_t, i64)
  DEFINE_UNARY_OP(abs, optional<int64_t>, optional<int64_t>, i64_opt)


  

  }  // namespace lite
}  // namespace cudf
