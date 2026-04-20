Add POC:

 - [x] Successfully compile existing CUDF JIT kernels whilst reducing the binary size of the JIT kernels
 - [x] Precompiled and JIT-ed compute_columns_kernel; reduce register pressure and ...:
    - [ ] Make AST codegen to use the pre-compiled operators and element_storage types
 - [ ] Manage the AOT-compiled artifacts with keys for the system
 - [x] Functions to compile the artifacts and embed them
 - [ ] Add sample pre-compiled UDF fragments
 - [ ] Build matrix tables that can be used for lookup/testing to check existence of fragments - [ ] Store mangled identifier names of the fragments

