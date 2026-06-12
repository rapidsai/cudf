# Libcudf Examples

This folder contains examples to demonstrate libcudf use cases. Running `build.sh` builds all
libcudf examples.

Current examples:

- Basic: demonstrates a basic use case with libcudf and building a custom application with libcudf
- Strings: demonstrates using libcudf for accessing and creating strings columns and for building custom kernels for strings
- Nested Types: demonstrates using libcudf for some operations on nested types
- Variant Workload: exercises `cudf::io::parquet::extract_variant_field` on a nested-VARIANT
  field-extraction workload (57 JSONPath-like extractions against three synthesized VARIANT
  columns) and reports end-to-end throughput
