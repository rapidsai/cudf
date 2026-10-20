# CUDA row operators

`cudf::transform`, `cudf::transform_lto`, and
`cudf::filter_join_indices_jit` execute through precompiled, type-erased row
kernels. Their `row_operator` parameter defaults to `true`. RowIR emits only
this ABI. CUDA value UDFs, PTX UDFs, and value LTO fragments use adapters to the
same row entries; set `row_operator=false` to select the value convention
explicitly. CUDA source with an existing value UDF signature remains accepted
by the allocating transform and join APIs.

## Calling convention

Source row operators have `cudf/row_abi.cuh` and its namespace available
automatically. An increment with a checked arithmetic operation is:

```cuda
#include <cudf/row_operators.cuh>
extern "C" __device__ status transform_operator(
    void const* inputs, void const* outputs, void* user_data, i32 row) {
  auto value = load_i32_nullable(inputs, 0, row);
  set_validity(outputs, 0, row, value.validity);
  if (!value.validity) { return success; }
  auto incremented = cudf_row_op_ADD_OVERFLOW_i32_i32(value.value, 1);
  if (incremented.error != success) { return incremented.error; }
  return set_i32(outputs, 0, row, incremented.value);
}
```

Pass this source to `transform` with `udf_source_type::CUDA`. Outputs may be
allocation specifications or a `mutable_table_view` of caller-prepared storage.
The operator owns column access; pointers to the column arrays and individual
columns are opaque and must be passed only to ABI access functions. Input
scalars occupy a single row: load row zero for them. Other inputs match the
output row count.

A join predicate uses the boolean variant of the transform entry:

```cuda
extern "C" __device__ status predicate_operator(
    void const* columns, bool8* outputs, void* user_data, i32 row) {
  auto left = load_i32_nullable(columns, 0, row);
  auto right = load_i32_nullable(columns, 1, row);
  *outputs = left.validity && right.validity && left.value > right.value;
  return success;
}
```

Inputs from both tables are gathered into aligned pair rows, ordered left then
right. Gathering preserves the input schema’s physical string-offset and
dictionary-index widths so the specialized loaders remain valid. Scalars still
use row zero. The predicate receives the same four arguments as the transform,
returning `status` and writing its boolean result through the `bool8* outputs`
parameter. Return an arithmetic failure status when evaluation fails.
Null predicates are false; unmatched-map sentinels bypass the operator. INNER, LEFT,
and FULL joins retain the existing join-filter postprocessing. Non-success
statuses raise `evaluation_error` before results are returned.

Every invocation receives `user_data`, even when it is null. Calls can have
side effects and execute once per participating row. Keep column storage,
borrowed string data, and user data alive on the supplied stream.

ABI version 1 uses natural CUDA alignment, 64-bit device pointers, fundamental
integer types, and `optional<T>{value, validity}`. A zero validity means the
payload must not be read. Decimal values carry their scale; timestamp and
duration aggregates carry the count in their declared unit. These are element
representations. `dictionary_index` is an alias for `i32` in `abi_types.cuh`.

## Access families and prepared output storage

Supported families are `T`, `list<T>`, `dictionary<T>`, and
`dictionary<list<T>>`, where T is a non-nested fixed-width or string type.
Structs are unsupported. Payload loaders return T directly:

```cuda
i32 plain = load_i32(inputs, column_index, row);
i32 list = load_list_i32(column, row, element);
i32 dictionary = load_dict_i32(inputs, column_index, row, index_type_tag<u8>{});
i32 dictionary_list = load_dict_list_i32(column, row, element, index_type_tag<i16>{});
string_view text = load_string_view(inputs, column_index, row, type_tag<i64>{});
```

Obtain an opaque column with `cudf_row_input_column(inputs, index)` or
`cudf_row_output_column(outputs, index)`. Dictionary overloads take an explicit
`index_type_tag` for their physical index width (all eight signed and unsigned
integer types are supported). String overloads take `type_tag<i32>` or
`type_tag<i64>` for their physical offsets, after the index tag for dictionary
strings. The caller selects these tags from the host schema before compilation.
Access functions never dispatch on device type IDs. RowIR selects tags from the
input representation and omits mask access for nonnullable inputs.

Each family also has `_nullable` loaders returning `optional<T>`. Setters accept
plain payloads; validity is written separately with `set_validity`. For example,
`load_i32_nullable`, `load_list_i32_nullable`,
`load_dict_i32_nullable`, and `load_dict_list_i32_nullable` include mask handling.
Plain accessors omit mask access. All variants select codecs statically.
Nullable dictionary loads check the parent and selected key; nullable list loads
check the parent and child before reading payloads. Nullable list sizes use
`load_list_size_nullable` and `load_dict_list_size_nullable`, returning `optional<i32>`.
A null parent bypasses offsets and indices, so its unused payload need not be valid.
Plain access requires valid, in-range rows and elements.

Prepare output buffers before calling the overload taking mutable views:

- `set_T(columns, index, row, T)` and `set_list_T(column, row, element, T)` write
  plain payloads. Decimal scales must match prepared storage, and list element
  indices must be in range. Plain setters do not change validity.
- String setters take the corresponding offset tag and copy bytes into prepared
  character storage if the byte count matches. `string_data(column, row, tag)`
  and `string_size(column, row, tag)` expose the same storage for direct writes.
  The untagged `set_string_view` overload stores borrowed views into an allocating
  transform’s view buffer; tagged overloads copy bytes into prepared character storage.
  The overload is selected when generating or compiling the operator.
- `set_dict_T` and `set_dict_list_T` take
  `(columns, index, row, dictionary_index, index_type_tag<I>)`, with an additional
  offset tag for string T. They select an existing immutable key, including a
  whole list key. Indices must be in range and representable by I.
- `set_validity(columns, index, row, validity)` updates a parent mask cooperatively
  within each warp. Its `(columns, index, row, element, validity)` overload updates a
  list child's mask, preserving parent validity; null parents bypass child access.
  These functions do not read or write payloads. Only call payload setters for valid
  values. Dictionary setters take `dictionary_index` and preserve immutable keys.
  Prepare a mask for any output or child that can become null.


Null-unaware prepared transforms skip rows with null inputs and propagate those
nulls to output masks; prepare parent masks when input nulls are possible.

Operators may change values and validity while preserving string/list
addresses and dimensions. They cannot construct offsets, allocate child
storage, or extend dictionary keys. Allocate masks for every output or child
that can become null. Prepared-storage requirements are preconditions; only
arithmetic errors use the evaluation status. After validity
changes, recompute owning-column null counts: a mutable view cannot update its
owner's cached metadata. If null strings or lists retain non-empty regions,
normalize the resulting owning column with `cudf::purge_nonempty_nulls` before
using APIs that require empty nulls. Sliced views preserve mask bits outside
their interval.

## Compilation and reuse

`abi_types.cuh`, `row_abi.cuh`, and `row_operators.cuh` are dependency-free.
AST and source row fragments use the existing `get_kernel_fragment` compilation
path. The shared wrapper includes `row_abi.cuh` before `nv_hdrstop` and the
operator source afterward, allowing the ABI declarations to participate in PCH.
`expected<T, E>{value, error}` lives in `abi_types.cuh` and has no optional
member. Infallible value operators take and return plain payloads; fallible value
operators return `expected<T, status>`. Only operators whose semantics depend on nulls take
`optional<T>` arguments. Their results are plain when always valid (for example,
`IS_NULL`), and `optional<T>` when they can be null (for example, `COALESCE`).
RowIR guards value operators with input validity and handles arithmetic status
separately. AST PROPAGATE reports errors; NULLIFY produces a null element, which is false as a join predicate.

For external code, include `cudf/row_abi.cuh` and use its namespace explicitly.
Define exported functions with `__device__ __inline_hint__` to permit inlining
across LTO modules while retaining the device symbols. Compile a compatible
CUDA LTO fragment, for example:

```sh
nvcc --fatbin --std=c++20 \
  --generate-code=arch=compute_75,code=lto_75 \
  -I cpp/include increment.cu -o increment.fatbin
```

Pass bytes defining `transform_operator` to `transform_lto` in its default row
mode. Source operators can link separately compiled helpers through the
`dependencies` parameter of the prepared transform and join APIs, using
`lto_fragment{bytes, lto_binary_type::FATBIN}`. A predicate entry itself may be
provided by a dependency when the CUDA source declares its prototype. Keep
fragment bytes alive until compilation returns. Fragments must share the ABI
version and use a compatible CUDA toolchain. Runtime source compilation uses
the JIT header bundle and the normal PCH-enabled fragment compilation path.

Cache identity includes source, ABI headers, representation schemas, compiler
and linker versions, compilation options, target architecture, and the bytes
and binary kind of every dependency, including named fragments. Buffer pointers,
row counts, and slice offsets do not specialize generated programs. Reusable
transform programs retain their kernels and AST literal ownership.

Concrete operator definitions remain in `operators.cu`. The build compiles each
operator signature and accessor overload into its own LTO fragment. RowIR
collects exact symbol dependencies during lowering and links only those
fragments, including output setters, validity, index/offset tags, and implicit
predicate operations. Opaque user code uses complete fallback fragments. Unary, binary, and ternary macros share their definitions;
each operator applies its signature to a type family. Loader/setter macros emit
the concrete access wrappers and declarations, including statically selected dictionary index and string offset widths.

When changing the RowIR operator/type table, update the corresponding macro
instantiations in `operators.cu` and declarations in `row_operators.cuh`. Shared
signature macros are defined in `operator_macros.cuh`.
