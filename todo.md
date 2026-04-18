Add POC:
 - Successfully compile existing CUDF JIT kernels whilst reducing the binary size of the JIT kernels
 - Precompiled and JIT-ed compute_columns_kernel; reduce register pressure and ...:
    - Make AST codegen to use the pre-compiled operators and element_storage types
 - Manage the AOT-compiled artifacts with keys for the system
 - Functions to compile the artifacts and embed them











