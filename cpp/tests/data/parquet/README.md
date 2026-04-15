# Parquet test fixtures

## VARIANT

### Unshredded (generator + libcudf tests)

- `variant_minimal.parquet` — one row, column `v` only, object `{ "x": 7 }`. Used by `parquet_variant_roundtrip_test.cpp`.
- `variant_multirow.parquet` — three rows, different metadata dictionaries per row; objects `{ "x", "k" }`, `{ "x", "y" }`, `{ "k" }` with INT32 and short STRING values. Same tests as above.

### DuckDB-produced sample

- `duckdb_variant_sample.parquet` — two rows, columns `id` (INT32) and `v` (VARIANT). Logical rows: `(42, { "k": "hello", "n": 42 })` and `(43, { "k": "world", "n": 0 })` when read with **DuckDB**. Physical schema is DuckDB's VARIANT Parquet layout (group with `metadata` / `value` BYTE_ARRAY plus typed child columns); **PyArrow** often fails to open this file (`Logical type Variant cannot be applied to group node`). Not used by libcudf gtests today—keep for cross-tool checks and future reader work.

### Regenerating

From the repository root (requires **PyArrow**):

```bash
python3 cpp/tests/scripts/parquet_variant_fixture_gen.py
```

The generator writes struct columns with PyArrow, then patches the Thrift footer to set **Parquet logical type `VARIANT`** on the parent group `v` (union field 16), matching what libcudf's reader expects for `PARQUET_COLUMN_BUFFER_FLAG_VARIANT_BINARY` on the **unshredded** `metadata` / `value` pair only.

### DuckDB cross-check (optional)

If the `duckdb` CLI or Python package is available, inspect schema or read VARIANT values (DuckDB decodes them as structs / maps in result rows):

```sql
FROM parquet_schema('variant_minimal.parquet');
PRAGMA parquet_metadata('variant_minimal.parquet');
```

### External fixtures (Spark / Java)

Place additional Spark-produced unshredded VARIANT files under this directory (or a sibling `spark/` subfolder) and reference them from new tests; note the **tool and version** in the commit message or here.
