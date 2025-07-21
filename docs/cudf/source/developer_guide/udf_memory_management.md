# String UDF memory management

Some functions used on strings inside UDFs create new strings; these include
functions like ``concat()``, ``replace()``, and various others. For a CUDA thread
to create a new string, it must dynamically allocate memory on the device to hold
the string's data. The cleanup of this memory by the thread later on must preserve
python's semantics, for example when the variable corresponding to the new string
goes out of scope. To accomplish this in cuDF, UDF memory management (allocation
and freeing of the underlying data) is handled transparently for the user, via a
reference counting mechanism. Along with the code generated from the functions and
operations within the passed UDF, numba-cuda will automatically weave the necessary
reference counting operations into the final device function that each thread will
ultimately run. This allows the programmer to pass a UDF that may utilize memory
allocating types such strings generally as one would in python:

```python
def udf(string):
if len(string) > 2:
  result = string.upper() # new allocation
else:
  result = string + string # new allocation
return result + 'abc'
```


## Numba memory management and the Numba Runtime (NRT)

The API functions used to update the reference count associated with a variable
derive from [Numba's memory management for nopython mode
code](https://numba.readthedocs.io/en/stable/developer/numba-runtime.html#memory-management).
This runtime library (NRT or Numba Runtime) provides implementations for operators
that increase and decrease a variable's reference count (INCREF/DECREF), and numba
analyzes the passed UDF to determine where the calls targeting these implementations
should go and what objects they should operate on. Below are some  examples of situations
where numba-cuda would detect a reference counting operation needs to be applied to
an object:

- **The creation of a new object**: During object creation, memory is allocated
and a structure to track the memory is created and initialized.
- **When new references are created:** For example during assignments, the
reference count of the assigned object is incremented.
- **When references are destroyed:** For example when an object goes out of
scope, or when an object holding a reference is destroyed. During these
events, the reference count of the tracked object is decremented. If the
reference count of an object falls to zero, its destructor will be invoked via
the Numba Runtime.
- **When an intermediate variable is no longer needed** For example when creating
a new variable for inspection then disposing of it, as in `string.upper() == 'A'`


Numba does not reference count every variable, as only variables with an associated
heap memory allocation need to be tracked. Numba determines if this is true for a
variable during compilation by querying the properties of the datamodel underlying
the variable's type. We provide a string type ``ManagedUDFString`` that implements
the required properties and backs any new string that is created on the device.


## Data structures
The core concept is a ``ManagedUDFString`` numba extension type that fulfills the
requirements to be reference counted by NRT. It is composed of a ``cudf::udf_string``
that owns its data and a pointer to a ``MemInfo`` that owns that string.

```python
@register_model(ManagedUDFString)
class managed_udf_string_model(models.StructModel):
  _members = (("meminfo", types.voidptr), ("udf_string", udf_string))

  def __init__(self, dmm, fe_type):
      super().__init__(dmm, fe_type, self._members)

  def has_nrt_meminfo(self):
      return True

  def get_nrt_meminfo(self, builder, value):
      # effectively returns self.meminfo in IR form
      udf_str_and_meminfo = cgutils.create_struct_proxy(managed_udf_string)(
          cuda_target.target_context, builder, value=value
      )
      return udf_str_and_meminfo.meminfo
```

The actual NRT APIs for adjusting the reference count of an object expect to operate
on a small struct called a ``MemInfo`` that _leads_ to the true object and also holds
the reference count:

```c++
extern "C"
struct MemInfo {
cuda::atomic<size_t, cuda::thread_scope_device> refct;
NRT_dtor_function dtor;
void* dtor_info;
void* data;
size_t size;
};
typedef struct MemInfo NRT_MemInfo;
```

Every instance of a reference counted type within the scope of a CUDA thread executing
the UDF is associated with a separate instance of this ``MemInfo`` struct. An INCREF or
DECREF on the instance in numba's intermediate representation formed during compilation
will resolve to an increase or decrease of the `refct` of the ``MemInfo`` associated
with that instance. The NRT_decref implementation calls the ``dtor`` on the ``data`` if
the ``refct`` is found to be zero.

The implementation of operations such as ``concat`` that return a reference counted type
is also responsible for constructing and initializing a new ``MemInfo`` associated with
the new instance.

## NRT Requirements

1. The datamodel must return ``True`` for the method ``has_nrt_meminfo()``
2. The datamodel must implement a method ``get_nrt_meminfo`` which produces a pointer
  to a `MemInfo`, a small struct that will be operated on

``ManagedUDFString`` fulfills (2) by tying the MemInfo and the object it tracks
together in a single managed type that simply returns its own ``.meminfo`` member when
queried.


### cuDF string data structures

On the C++ side, strings are represented using the ``cudf::string_view`` class,
and the ``cudf::strings::udf::udf_string`` class. The ``string_view`` class does
not own the underlying string data; the ``udf_string`` does own the underlying
string data. The ``cudf::string_view`` class inherits from libcudf, where it is
useful for computng read-only functions of a single string, for a single thread,
such as finding it's length. However ``cudf::udf_string`` is udf specific and is
the type that implements algorithms that create new strings for a single thread,
such as ``concat``. In classic libcudf, this object is not needed, and operations
like ``concat`` are implemented in a whole column sense, eliminating the need for
this abstraction.

The cuDF extensions to Numba generate code to manipulate instances of these
classes, so we outline the members of these classes to aid in understanding
them. These classes also have various methods; consult the cuDF C++ Developer
Documentation for further details of these structures.

```c++
class string_view {
  // A pointer to the underlying string data
  char const* p{};
  // The length of the underlying string data in bytes
  size_type bytes{};
  // The offset into the underlying string data in characters
  size_type char_pos{};
  // The offset into the underlying string data in bytes
  size_type byte_pos{};
};
```

```c++
class udf_string {
// A pointer to the underlying string data
char* m_data{};
// The length of the string data in bytes
cudf::size_type m_bytes{};
// The size of the underlying allocation in bytes
cudf::size_type m_capacity{};
};
```

```{note}
A ``udf_string`` has a destructor that frees the underlying string data. This is
important, because the C++ destructor is invoked during destruction of a
Python-side Managed UDF String object.
```


## Implementation

The cuDF implementations for Managed UDF Strings is required to provide:

- Typing and lowering for Managed UDF String operations.
- The typing has no special properties; it is similar to any other typing
  implementation in a Numba extension.
- The lowering is required to ensure that ``NRT_MemInfo`` objects for each
  managed object are created and initialized correctly.
- C++ implementations of string functions:
- Some of these are provided by cuDF / libcudf's C++ string functionality
- Other functions are provided by the ``strings_udf`` C++ library in cuDF
  Python. These help with the allocation of data and implement the required
  destructors.
- Numba shim functions to adapt calls to C++ code for use in Numba code and
  Numba extensions are also required.
- Conversion from String UDF data back into cudf C++ column format.

Use of C++ code for string functionality is not a hard requirement for
implementing string support in a Numba extension - it is instead a pragmatic
requirement that the Python and C++ sides of cuDF share a single implementation
for string operations, instead of trying to keep two separately-maintained
implementations in sync.

cuDF generally does not need to provide any implementation to support reference
counting operations during the lifetime of a managed object after its
construction. Because the standard ``NRT_MemInfo`` data structure is used in
conjunction with Numba's built-in lowerings and NRT library, it automatically
emits code to manage reference counts during "normal" Python code execution
including assignments, scoping, function calls, etc. Another way to view this is
that the cuDF code is really providing the implementation and details at the
lifetime boundaries, the beginning and ending, of Managed UDF Strings.

The majority of the complexity in the implementation comes from two areas:

- Combining the requirement to use C++ implementations, with the need to provide
correct initialization of `NRT_MemInfo` object, and
- Conversion of Managed UDF String objects back into cuDF columns when a UDF
returns strings.


## String Lifecycle Details

Let's trace the complete lifecycle of a string created by `result = str1 + str2` in a UDF:

#### Phase 1: Compilation

**1.1 Numba Analysis**
```python
# User UDF
def my_udf(str1, str2):
    result = str1 + str2
    return result
```

- Numba identifies `str1 + str2` as returning a `ManagedUDFString`
- Lowering phase begins for the `+` operator

**1.2 Stack Allocation**
```python
managed_ptr = builder.alloca(
    context.data_model_manager[managed_udf_string].get_value_type()
)
```
- Allocates stack space for the complete `ManagedUDFString` struct
- At this point, both fields are uninitialized

**1.3 Member Pointer Extraction**
```python
udf_str_ptr = builder.gep(managed_ptr, [ir.IntType(32)(0), ir.IntType(32)(1)])
```
- Gets pointer to the `udf_string` member within the allocated struct

#### Phase 2: String Creation via Shim Function

**2.1 Shim Function Call**
```python
meminfo = context.compile_internal(
    builder, call_concat_string_view,
    types.voidptr(_UDF_STRING_PTR, _STR_VIEW_PTR, _STR_VIEW_PTR),
    (udf_str_ptr, lhs_ptr, rhs_ptr)
)
```

**2.2 Inside the Shim Function**
```c++
extern "C" __device__ int concat_shim(void** out_meminfo,
                                      void* output_udf_str,
                                      void* const* lhs,
                                      void* const* rhs) {
    auto lhs_sv = reinterpret_cast<cudf::string_view const*>(lhs);
    auto rhs_sv = reinterpret_cast<cudf::string_view const*>(rhs);

    // Perform actual concat- allocates GPU memory for result
    auto result_str = cudf::strings::udf::concat(*lhs_sv, *rhs_sv);

    // Place result into pre-allocated stack space using placement new
    auto udf_str_ptr = new (output_udf_str) udf_string(std::move(result_str));

    // Create and return the meminfo
    *out_meminfo = make_meminfo_for_new_udf_string(udf_str_ptr);

    return 0;
}
```

**2.3 MemInfo Creation Detail**
```c++
__device__ NRT_MemInfo* make_meminfo_for_new_udf_string(udf_string* udf_str) {
    struct mi_str_allocation {
        NRT_MemInfo mi;
        udf_string st;
    };

    // Single heap allocation for both structures
    mi_str_allocation* heap_allocation = (mi_str_allocation*)NRT_Allocate(sizeof(mi_str_allocation));

    NRT_MemInfo* mi_ptr = &(heap_allocation->mi);
    udf_string* heap_str_ptr = &(heap_allocation->st);

    // Initialize MemInfo pointing to co-allocated string
    NRT_MemInfo_init(mi_ptr, heap_str_ptr, 0, udf_str_dtor, NULL);

    // Copy string data to heap location
    memcpy(heap_str_ptr, udf_str, sizeof(udf_string));

    return mi_ptr;
}
```

#### Phase 3: Object Assembly and Return

**3.1 Final Assembly**
```python
managed = cgutils.create_struct_proxy(managed_udf_string)(context, builder)
managed.meminfo = meminfo  # Points to heap MemInfo
return managed._getvalue()
```

**3.2 Current Memory State**
- **Stack**: `ManagedUDFString` struct with valid `meminfo` pointer and `udf_string` data
- **Heap**: Co-allocated MemInfo and udf_string structures
- **GPU Memory**: String data owned by heap-allocated udf_string
- **Reference Count**: 1 (object just created)

#### Phase 4: Runtime Usage and Reference Management

**4.1 Assignment Operations**
```python
# When user does: another_var = result
```
- Numba detects assignment to reference-counted type
- Automatically inserts `NRT_incref(managed.meminfo)`
- `heap_allocation->mi.refct` becomes 2
- Both `result` and `another_var` point to same MemInfo

**4.2 Scope Changes**
```python
# When result goes out of scope or function returns
```
- Numba automatically inserts `NRT_decref(managed.meminfo)`
- `heap_allocation->mi.refct` decrements to 1
- Object remains alive because refct > 0

**4.3 Function Calls**
```python
# When passing to another function: some_func(result)
```
- Numba may insert temporary incref/decref pairs
- Ensures object stays alive during parameter passing
- Reference count fluctuates but returns to stable state

#### Phase 5: Destruction Sequence

**5.1 Final Reference Release**
```python
# When last reference goes out of scope
```
- Numba inserts final `NRT_decref(managed.meminfo)`
- `heap_allocation->mi.refct` becomes 0
- NRT automatically calls `mi.dtor(mi.data, mi.size, mi.dtor_info)`

**5.2 Destructor Execution**
```c++
__device__ void udf_str_dtor(void* udf_str, size_t size, void* dtor_info) {
    auto ptr = reinterpret_cast<udf_string*>(udf_str);
    ptr->~udf_string();
}
```

The destructor is called with `udf_str` pointing directly to the heap-allocated `udf_string`. The C++ destructor automatically frees the GPU memory containing the actual string data. The NRT system handles freeing the co-allocated MemInfo block separately.


**5.3 Final Memory State**
- **GPU String Memory**: Freed (the actual "helloworld" data)
- **Heap MemInfo Block**: Freed (the 64-byte management structure)
- **Stack**: Original `ManagedUDFString` becomes invalid/out-of-scope
- **Reference Count**: N/A (object destroyed)
