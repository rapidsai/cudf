-- Optional sanity checks for VARIANT Parquet fixtures (schema / physical layout).
-- Run: duckdb -c ".read cpp/tests/scripts/duckdb_inspect_variant_parquet.sql"
-- Adjust paths if needed.

FROM parquet_schema('cpp/tests/data/parquet/variant_minimal.parquet');
FROM parquet_schema('cpp/tests/data/parquet/variant_multirow.parquet');
