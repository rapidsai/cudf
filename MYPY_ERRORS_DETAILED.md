# Mypy Remaining Errors - Detailed Analysis

**Date**: Updated as of commit 13322be2f2
**Total Errors**: 67 errors in 23 files (checked 468 source files)

## Progress Summary

### ✅ Completed Categories
- **Category 1**: Pandas APIs Missing from Stubs (completed earlier)
- **Category 2**: ExtensionDtype Issues (completed earlier)
- **Category 3**: Dtype Parameter Type Mismatches (completed earlier)
- **Category 4**: Timestamp/Timedelta Issues (completed earlier)
- **Category 5** (Partial): Method Override Violations (completed earlier)
- **Category 6**: _with_type_metadata Override Issues - 11 errors (commit ca699df766)
- **Category 7**: ExtensionDtype/Custom Dtype Attributes - 12 errors (commits 9dc42971f0, 4e02a32f06, fdc18a4cf8, 0c934c0c25)
- **Category 8**: ExtensionDtype.itemsize Access - 10 errors (commit 38a7405b0b)
- **Category 9**: ExtensionDtype.str Access - 1 error (commit 9f619dddaa)
- **Category 10**: ExtensionDtype.fields Access - 1 error (commit f181c8b462)

**Total Fixed**: 24 errors in categories 7-10

### ⏳ Remaining: 67 errors

**Categories 11-16** require more careful analysis rather than mechanical fixes.

---

## Category 11: Type Assignment Issues (10 errors)

### 11.1 _PD_SCALAR type inference (3 errors)

**Error 1**: `timedelta.py:68`
```
error: Incompatible types in assignment (expression has type "type[Timedelta]", base class "TemporalBaseColumn" defined the type as "Timestamp | Timedelta")  [assignment]
```

**Error 2**: `datetime.py:90`
```
error: Incompatible types in assignment (expression has type "type[Timestamp]", base class "TemporalBaseColumn" defined the type as "Timestamp | Timedelta")  [assignment]
```

**Error 3**: `timedelta.py:282` & `datetime.py:166`
```
error: Cannot determine type of "_PD_SCALAR"  [has-type]
```

**Fix**: Add explicit type annotations to `_PD_SCALAR` class variables

---

### 11.2 Index/MultiIndex assignments (2 errors)

**Error 1**: `column_accessor.py:322`
```
error: Incompatible types in assignment (expression has type "Index[Any]", variable has type "MultiIndex")  [assignment]
```

**Error 2**: `dataframe.py:7603`
```
error: Incompatible types in assignment (expression has type "Index[Any]", variable has type "MultiIndex")  [assignment]
```

**Fix**: Add isinstance check or type narrowing

---

### 11.3 Dtype assignments with None (5 errors)

**Error 1**: `column.py:2419`
```
error: Incompatible types in assignment (expression has type "ExtensionDtype | dtype[Any] | None", variable has type "ExtensionDtype | dtype[Any]")  [assignment]
```

**Error 2**: `numerical.py:580`
```
error: Incompatible types in assignment (expression has type "ExtensionDtype | str | dtype[Any]", variable has type "ExtensionDtype | dtype[Any]")  [assignment]
```

**Error 3**: `numerical.py:592`
```
error: Incompatible types in assignment (expression has type "ExtensionDtype | str | dtype[Any]", variable has type "ExtensionDtype | dtype[Any]")  [assignment]
```

**Error 4**: `indexed_frame.py:6970`
```
error: Incompatible types in assignment (expression has type "ExtensionDtype | dtype[Any] | None", variable has type "ExtensionDtype | dtype[Any]")  [assignment]
```

**Error 5**: `frame.py:632`
```
error: Incompatible types in assignment (expression has type "ExtensionDtype | dtype[Any] | None", variable has type "ExtensionDtype | dtype[Any]")  [assignment]
```

**Error 6**: `index.py:5215`
```
error: Incompatible types in assignment (expression has type "dtype[Any] | ExtensionDtype | None", variable has type "ExtensionDtype | str | dtype[Any]")  [assignment]
```

**Fix**: Update variable type annotations to accept None or str

---

### 11.4 DataFrame/Series assignment (1 error)

**Location**: `io/json.py:436`
```
error: Incompatible types in assignment (expression has type "DataFrame | Series[Any]", variable has type "DataFrame")  [assignment]
```

**Fix**: Type narrowing or updated annotation

---

### 11.5 DateOffset assignment (1 error)

**Location**: `tools/datetimes.py:870`
```
error: Incompatible types in assignment (expression has type "pandas._libs.tslibs.offsets.DateOffset", variable has type "cudf.core.tools.datetimes.DateOffset")  [assignment]
```

**Fix**: Type narrowing or alias

---

### 11.6 IntervalIndex assignment (1 error)

**Location**: `core/cut.py:243`
```
error: Incompatible types in assignment (expression has type "Any | None", variable has type "IntervalIndex[Any]")  [assignment]
```

**Fix**: Type narrowing

---

## Category 12: Function Argument Type Issues (13 errors)

### 12.1 Dtype union mismatches (8 errors)

**Error 1**: `column.py:114`
```
error: Argument 1 to "_can_values_be_equal" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any]"  [arg-type]
```

**Error 2**: `column.py:116`
```
error: Argument 2 to "_can_values_be_equal" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any]"  [arg-type]
```

**Error 3**: `column.py:743` (2 errors)
```
error: Argument 1 to "get" of "dict" has incompatible type "ExtensionDtype | dtype[Any]"; expected "dtype[Any]"  [arg-type]
error: Argument 2 to "get" of "dict" has incompatible type "ExtensionDtype | dtype[Any]"; expected "ExtensionDtype"  [arg-type]
```

**Error 4**: `column.py:2517`
```
error: Argument 1 to "_with_type_metadata" of "ColumnBase" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any] | None"  [arg-type]
```
*Note: This error was revealed after fixing Category 6*

**Error 5**: `numerical.py:303`
```
error: Argument 1 to "get_dtype_of_same_kind" has incompatible type "ExtensionDtype | dtype[Any] | None"; expected "ExtensionDtype | str | dtype[Any]"  [arg-type]
```

**Error 6**: `numerical.py:591`
```
error: Argument "dtype" to "cast" of "ColumnBase" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any]"  [arg-type]
note: Error code "arg-type" not covered by "type: ignore" comment
```

**Error 7**: `numerical.py:595`
```
error: Argument "dtype" to "cast" of "ColumnBase" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any]"  [arg-type]
note: Error code "arg-type" not covered by "type: ignore" comment
```

**Error 8**: `single_column_frame.py:54`
```
error: Argument 1 to "is_dtype_obj_numeric" has incompatible type "ExtensionDtype | str | dtype[Any]"; expected "ExtensionDtype | dtype[Any]"  [arg-type]
```

**Fix**: Type narrowing to exclude `str` or `None` from dtype unions before passing to functions

---

### 12.2 DatetimeTZDtype argument mismatch (3 errors)

**Error 1**: `datetime.py:837`
```
error: Argument 1 to "_get_base_dtype" has incompatible type "ExtensionDtype | dtype[Any]"; expected "DatetimeTZDtype"  [arg-type]
```

**Error 2**: `datetime.py:849`
```
error: Argument 1 to "_get_base_dtype" has incompatible type "ExtensionDtype | dtype[Any]"; expected "DatetimeTZDtype"  [arg-type]
```

**Error 3**: `datetime.py:872`
```
error: Argument 1 to "as_datetime_column" of "DatetimeColumn" has incompatible type "dtype[Any] | DatetimeTZDtype"; expected "dtype[Any]"  [arg-type]
```

**Fix**: Update function signatures to accept DatetimeTZDtype or add type narrowing

---

### 12.3 find_common_type with None (2 errors)

**Error 1**: `frame.py:1435`
```
error: Argument 2 has incompatible type "ExtensionDtype | dtype[Any] | None"; expected "ExtensionDtype | str | dtype[generic[Any]] | type[str] | type[complex] | type[bool] | type[object]"  [arg-type]
```

**Error 2**: `frame.py:1445`
```
error: Argument 2 has incompatible type "ExtensionDtype | dtype[Any] | None"; expected "ExtensionDtype | str | dtype[generic[Any]] | type[str] | type[complex] | type[bool] | type[object]"  [arg-type]
```

**Fix**: Add None checks before calling

---

### 12.4 Additional argument type issues (3 errors)

**Error 1**: `dataframe.py:6654`
```
error: Argument 1 to "find_common_type" has incompatible type "tuple[ExtensionDtype | dtype[Any] | None, dtype[Any]]"; expected "Iterable[ExtensionDtype | dtype[Any]]"  [arg-type]
```

**Error 2**: `dataframe.py:6681`
```
error: Argument 1 to "get_dtype_of_same_kind" has incompatible type "ExtensionDtype | dtype[Any] | None"; expected "ExtensionDtype | str | dtype[Any]"  [arg-type]
```

**Error 3**: `dataframe.py:7562`
```
error: Argument 2 to "column_empty" has incompatible type "ExtensionDtype | dtype[Any] | None"; expected "ExtensionDtype | str | dtype[Any]"  [arg-type]
```

**Fix**: Type narrowing to exclude None

---

## Category 13: Return Type Issues (8 errors)

### 13.1 List/Tuple return mismatch (1 error)

**Location**: `column_accessor.py:212`
```
error: Incompatible return value type (got "list[Hashable | None]", expected "tuple[Hashable, ...]")  [return-value]
```

**Fix**: Convert list to tuple: `return tuple(result)`

---

### 13.2 Dtype return narrowing (2 errors)

**Error 1**: `numerical.py:641`
```
error: Incompatible return value type (got "ExtensionDtype | dtype[Any]", expected "dtype[Any]")  [return-value]
```

**Error 2**: `numerical.py:662`
```
error: Incompatible return value type (got "ExtensionDtype | dtype[Any]", expected "dtype[Any]")  [return-value]
```

**Fix**: Type narrowing or update return annotation

---

### 13.3 Index subclass returns (2 errors)

**Error 1**: `index.py:2897`
```
error: Incompatible return value type (got "ExtensionDtype | dtype[Any]", expected "dtype[Any]")  [return-value]
```

**Error 2**: `index.py:4236`
```
error: Incompatible return value type (got "Index[Any]", expected "DatetimeIndex")  [return-value]
```

**Fix**: Type narrowing or cast

---

### 13.4 NumericalBaseColumn.median return (1 error)

**Location**: `numerical_base.py:235`
```
error: Incompatible return value type (got "ColumnBase | Any", expected "NumericalBaseColumn")  [return-value]
```

**Fix**: Update return annotation or add cast

---

### 13.5 _name type determination (1 error)

**Location**: `multiindex.py:443`
```
error: Cannot determine type of "_name"  [has-type]
```

**Fix**: Add explicit type annotation to `_name` class variable

---

### 13.6 from_dict return type (1 error)

**Location**: `dataframe.py:2458`
```
error: No overload variant of "from_dict" of "DataFrame" matches argument types "dict[Any, Any]", "str", "ExtensionDtype | str | dtype[Any] | None", "list[Any] | None"  [call-overload]
```

**Fix**: Type narrowing on dtype parameter or adjust call signature

---

## Category 14: Override Issues (2 errors)

### 14.1 __contains__ signature mismatch (1 error)

**Location**: `timedelta.py:130`
```
error: Signature of "__contains__" incompatible with supertype "TemporalBaseColumn"  [override]
note:      Superclass:
note:          def __contains__(self, datetime64[date | int | None] | timedelta64[timedelta | int | None], /) -> bool
note:      Subclass:
note:          def [DatetimeLikeScalar: (Period, Timestamp, Timedelta)] __contains__(self, DatetimeLikeScalar, /) -> bool
```

**Status**: Low priority, signature mismatch is intentional

---

### 14.2 element_indexing return type (1 error)

**Location**: `interval.py:157`
```
error: Return type "Interval[Any] | dict[Any, Any] | None" of "element_indexing" incompatible with return type "dict[Any, Any] | None" in supertype "StructColumn"  [override]
```

**Fix**: Update parent signature to allow Interval return or add type ignore

---

## Category 15: Complex Type Issues (11 errors)

### 15.1 PyArrow dict issues (3 errors)

**Error 1**: `column.py:916`
```
error: Dict entry 0 has incompatible type "None": "ChunkedArray[Any]"; expected "str": "list[Any] | Array[Any]"  [dict-item]
```

**Error 2**: `column.py:921`
```
error: Dict entry 0 has incompatible type "None": "Array[Any] | ChunkedArray[Any]"; expected "str": "list[Any] | Array[Any]"  [dict-item]
```

**Error 3**: `column.py:939`
```
error: Dict entry 0 has incompatible type "None": "ChunkedArray[Any]"; expected "str": "list[Any] | Array[Any]"  [dict-item]
```

**Fix**: Use empty dict {} or `# type: ignore[dict-item]`

---

### 15.2 ExtensionDtype call-arg error (1 error)

**Location**: `decimal.py:247`
```
error: Too many arguments for "ExtensionDtype"  [call-arg]
note: Error code "call-arg" not covered by "type: ignore" comment
```

**Fix**: Update type ignore comment to include call-arg

---

### 15.3 Operator issues with None (6 errors)

**Error 1**: `dataframe.py:2142`
```
error: Unsupported operand types for <= ("int" and "None")  [operator]
note: Right operand is of type "int | None"
```

**Error 2**: `dataframe.py:2159`
```
error: Unsupported operand types for + ("None" and "int")  [operator]
note: Left operand is of type "int | None"
```

**Error 3**: `dataframe.py:2167` (4 errors)
```
error: Unsupported operand types for > ("int" and "None")  [operator]
error: Unsupported operand types for < ("int" and "None")  [operator]
error: Unsupported operand types for / ("None" and "float")  [operator]
error: Unsupported operand types for % ("None" and "int")  [operator]
```

**Error 4**: `numerical.py:805`
```
error: Unsupported left operand type for <= ("ExtensionDtype")  [operator]
note: Both left and right operands are unions
```

**Fix**: Add None checks before arithmetic operations

---

### 15.4 Type variable issues (2 errors)

**Error 1**: `dataframe.py:2135`
```
error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "int | None"  [type-var]
```

**Error 2**: `dataframe.py:2159`
```
error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "int | None"  [type-var]
```

**Fix**: Filter out None values before calling max()

---

### 15.5 Type variable with dtype (2 errors)

**Error 1**: `join/_join_helpers.py:110`
```
error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "ExtensionDtype | dtype[Any]"  [type-var]
```

**Error 2**: `join/_join_helpers.py:117`
```
error: Value of type variable "SupportsRichComparisonT" of "max" cannot be "ExtensionDtype | dtype[Any]"  [type-var]
```

**Fix**: Type narrowing or use alternative comparison approach

---

## Category 16: Miscellaneous (12 errors)

### 16.1 Series constructor overload (1 error)

**Location**: `series.py:1962`
```
error: No overload variant of "Series" matches argument types "Any", "bool", "Any"  [call-overload]
note: Possible overload variants:
[... long list of overloads ...]
```

**Fix**: Adjust call signature or add `# type: ignore[call-overload]`

---

### 16.2 Value indexing on union (1 error)

**Location**: `series.py:2024`
```
error: Value of type "ExtensionDtype | str | dtype[Any] | dict[Hashable, ExtensionDtype | str | dtype[Any]]" is not indexable  [index]
```

**Fix**: Type narrowing before indexing

---

### 16.3 List comprehension type (1 error)

**Location**: `multiindex.py:1718`
```
error: List comprehension has incompatible type List[ndarray[Any, dtype[Any]] | Any]; expected List[Sequence[int]]  [misc]
```

**Fix**: Add explicit type annotations or casts

---

### 16.4 JSON compression type (1 error)

**Location**: `io/json.py:267`
```
error: Argument "compression" to "read_json" has incompatible type "Literal['bz2', 'gzip', 'infer', 'snappy', 'zip', 'zstd'] | None"; expected "Literal['infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd', 'tar'] | dict[str, Any] | None"  [arg-type]
```

**Fix**: `# type: ignore[arg-type]` - pandas-stubs literal mismatch

---

### 16.5 assert_frame_equal (1 error)

**Location**: `pandas/_benchmarks/utils.py:473`
```
error: Argument 1 to "assert_frame_equal" has incompatible type "Series[Any]"; expected "DataFrame"  [arg-type]
```

**Fix**: `# type: ignore[arg-type]` or adjust call

---

### 16.6 pandas-stubs missing attributes (3 errors)

**Error 1**: `api/extensions/accessor.py:5`
```
error: Module "pandas.core.accessor" has no attribute "CachedAccessor"  [attr-defined]
```

**Error 2**: `pandas/_wrappers/pandas.py:591`
```
error: Module "pandas.arrays" has no attribute "NumpyExtensionArray"  [attr-defined]
```

**Error 3**: `pandas/_wrappers/pandas.py:1184`
```
error: Module "pandas.io.formats.style" does not explicitly export attribute "StylerRenderer"  [attr-defined]
```

**Fix**: Add `# type: ignore[attr-defined]` for pandas-stubs gaps

---

### 16.7 Parquet I/O issues (2 errors)

**Error 1**: `io/parquet.py:1460`
```
error: "read_parquet" gets multiple values for keyword argument "engine"  [misc]
```

**Error 2**: `io/parquet.py:1460`
```
error: "read_parquet" gets multiple values for keyword argument "columns"  [misc]
```

**Fix**: Review function call and parameter passing

---

## Recommended Fix Order

1. ✅ ~~**Categories 7-10** (24 errors) - Mechanical fixes~~ **COMPLETED**
2. **Category 11** (10 errors) - Type assignment fixes - **NEXT**
3. **Category 12** (13 errors) - Function argument type fixes
4. **Category 13** (8 errors) - Return type fixes
5. **Category 16** (12 errors) - Miscellaneous issues (some are simple)
6. **Category 15** (11 errors) - Complex type issues (careful handling)
7. **Category 14** (2 errors) - Override issues (low priority)

**Estimated remaining effort**: ~3-5 hours
- Categories 11-13: ~2-3 hours (moderate complexity)
- Categories 15-16: ~2-3 hours (higher complexity)
- Category 14: ~0.5 hour (low priority)

**Progress**: 24/91 errors fixed (26% reduction). Categories 7-10 complete.
