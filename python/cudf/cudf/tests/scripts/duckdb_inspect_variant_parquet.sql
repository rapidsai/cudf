-- Optional sanity checks for VARIANT Parquet fixtures (schema / physical layout).
-- Run: duckdb -c ".read python/cudf/cudf/tests/scripts/duckdb_inspect_variant_parquet.sql"
-- Adjust paths if needed.

FROM parquet_schema('python/cudf/cudf/tests/data/parquet/variant_minimal.parquet');
FROM parquet_schema('python/cudf/cudf/tests/data/parquet/variant_multirow.parquet');
