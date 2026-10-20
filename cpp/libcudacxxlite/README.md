# libcudacxxlite

A lightweight subset of CCCL for cuDF's NVRTC JIT compilation, intended to reduce
header parsing and compilation time. It mirrors CCCL's include paths and provides
only the APIs needed by cuDF's JIT kernels. 

## Supported APIs

### `<cuda/atomic>`

- `cuda::atomic_ref<T, Scope>`

### `<cuda/bit>`

- `cuda::bitmask`

### `<cuda/cmath>`

- `cuda::ceil_div`

### `<cuda/numeric>`

- `cuda::overflow_result<T>`
- `cuda::add_overflow`
- `cuda::sub_overflow`
- `cuda::mul_overflow`
- `cuda::div_overflow`
- `cuda::mod_overflow`
- `cuda::neg_overflow`
- `cuda::abs_overflow`

### `<cuda/ptx>`

- `cuda::ptx::elect_sync` (SM90+)

### `<cuda/std/algorithm>`

- `min`
- `max`
- `clamp`

### `<cuda/std/atomic>`

- `cuda::memory_order` (also available in `cuda::std`)
- `cuda::memory_order_relaxed` (also available in `cuda::std`)
- `cuda::memory_order_consume` (also available in `cuda::std`)
- `cuda::memory_order_acquire` (also available in `cuda::std`)
- `cuda::memory_order_release` (also available in `cuda::std`)
- `cuda::memory_order_acq_rel` (also available in `cuda::std`)
- `cuda::memory_order_seq_cst` (also available in `cuda::std`)
- `cuda::thread_scope`
- `cuda::thread_scope_system`
- `cuda::thread_scope_device`
- `cuda::thread_scope_block`
- `cuda::thread_scope_thread`

### `<cuda/std/bit>`

- `bit_cast`
- `countl_zero`
- `countr_zero`
- `popcount`

### `<cuda/std/cassert>`

- `assert`

### `<cuda/std/chrono>`

- `duration`
- `duration_cast`
- `days`
- `hours`
- `minutes`
- `seconds`
- `milliseconds`
- `microseconds`
- `nanoseconds`
- `system_clock`
- `sys_time`
- `time_point`

### `<cuda/std/climits>`

- `CHAR_BIT`
- `INT_MAX`
- `INT_MIN`
- `UINT_MAX`
- `LONG_MAX`
- `LLONG_MAX`

### `<cuda/std/cmath>`

- `abs`
- `acos`
- `acosh`
- `asin`
- `asinh`
- `atan`
- `atanh`
- `cbrt`
- `ceil`
- `cos`
- `cosh`
- `exp`
- `fabs`
- `floor`
- `fmod`
- `log`
- `pow`
- `rint`
- `round`
- `sin`
- `sinh`
- `sqrt`
- `tan`
- `tanh`
- `trunc`

### `<cuda/std/concepts>`

- `convertible_to`
- `equality_comparable`
- `equality_comparable_with`
- `floating_point`
- `integral`
- `same_as`
- `totally_ordered`
- `totally_ordered_with`

### `<cuda/std/cstddef>`

- `size_t`
- `ptrdiff_t`
- `nullptr_t`
- `max_align_t`
- `byte`

### `<cuda/std/cstdint>`

- `int8_t`
- `int16_t`
- `int32_t`
- `int64_t`
- `uint8_t`
- `uint16_t`
- `uint32_t`
- `uint64_t`
- `int_fast8_t`
- `int_fast16_t`
- `int_fast32_t`
- `int_fast64_t`
- `uint_fast8_t`
- `uint_fast16_t`
- `uint_fast32_t`
- `uint_fast64_t`
- `int_least8_t`
- `int_least16_t`
- `int_least32_t`
- `int_least64_t`
- `uint_least8_t`
- `uint_least16_t`
- `uint_least32_t`
- `uint_least64_t`
- `intptr_t`
- `uintptr_t`
- `intmax_t`
- `uintmax_t`
- `::int8_t`
- `::int16_t`
- `::int32_t`
- `::int64_t`
- `::uint8_t`
- `::uint16_t`
- `::uint32_t`
- `::uint64_t`
- `::__int128_t`
- `::__uint128_t`

### `<cuda/std/expected>`

- `unexpected<E>`
- `expected<T, E>`

### `<cuda/std/functional>`

- `divides` (typed and transparent forms)
- `equal_to` (typed and transparent forms)
- `less` (typed and transparent forms)
- `minus` (typed and transparent forms)
- `multiplies` (typed and transparent forms)
- `plus` (typed and transparent forms)

### `<cuda/std/iterator>`

- `input_iterator_tag`
- `output_iterator_tag`
- `forward_iterator_tag`
- `bidirectional_iterator_tag`
- `random_access_iterator_tag`
- `iterator_traits`
- `distance` (subtraction only)

### `<cuda/std/limits>`

- `numeric_limits<T>` (integral types, float, double, and cv-qualified variants)

### `<cuda/std/optional>`

- `nullopt_t`
- `nullopt`
- `optional<T>`
- `operator==` (optional-to-optional)
- `operator!=` (optional-to-optional)

### `<cuda/std/ratio>`

- `rational<N, D>`
- `ratio<N, D>` (alias of rational)

### `<cuda/std/span>`

- `dynamic_extent`
- `span<T, Extent>`

### `<cuda/std/tuple>`

- `tuple`
- `tuple_element`
- `tuple_element_t`
- `get` (indexed)
- `apply`
- `tuple_cat`

### `<cuda/std/type_traits>`

- `integral_constant`
- `bool_constant`
- `true_type`
- `false_type`
- `is_same`
- `is_same_v`
- `is_integral`
- `is_integral_v`
- `is_floating_point`
- `is_floating_point_v`
- `is_arithmetic`
- `is_arithmetic_v`
- `is_signed`
- `is_signed_v`
- `is_unsigned`
- `is_unsigned_v`
- `is_void`
- `is_void_v`
- `is_convertible`
- `is_convertible_v`
- `disjunction`
- `disjunction_v`
- `remove_cv`
- `remove_cv_t`
- `remove_reference`
- `remove_reference_t`
- `make_signed`
- `make_signed_t`
- `make_unsigned`
- `make_unsigned_t`
- `conditional`
- `conditional_t`
- `enable_if`
- `enable_if_t`
- `common_type`
- `common_type_t`
- `remove_cvref_t`
- `decay_t`
- `void_t`
- `declval`

### `<cuda/std/utility>`

- `move`
- `forward`
- `swap`
- `integer_sequence`
- `make_integer_sequence`
- `index_sequence`
- `make_index_sequence`
- `index_sequence_for`
- `pair`
- `get` (indexed pair access)
