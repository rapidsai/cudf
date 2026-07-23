# cudf — Safe Rust API for GPU DataFrames

## Overview

The `cudf` crate provides a safe, idiomatic Rust interface to GPU-accelerated
DataFrame operations powered by NVIDIA cuDF. It wraps the raw FFI bindings in
`cudf-sys` with RAII ownership, `Result`-based error propagation, and type-safe
handles for columns, tables, and I/O.

Operations run on the GPU via CUDA and are coordinated through a `Resources`
handle that owns a CUDA stream.

## Prerequisites

`libcudf_c.so` must be installed before building this crate. Build it from the
`c/` directory at the repository root:

```bash
cmake -S c -B c/build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build c/build --parallel
cmake --install c/build --prefix $CONDA_PREFIX
```

The build scripts look for the library under `$CMAKE_PREFIX_PATH` /
`$CONDA_PREFIX`. Set one of those before running `cargo build`.

## Workspace Structure

```
rust/
├── Cargo.toml        # workspace manifest
├── cudf-sys/         # raw FFI bindings (auto-generated, checked in)
│   ├── build.rs      # links libcudf_c; optionally runs bindgen
│   └── src/
│       └── bindings.rs
└── cudf/             # safe high-level Rust API
    └── src/
        ├── lib.rs
        ├── resources.rs
        ├── column.rs
        ├── table.rs
        ├── binaryop.rs
        ├── join.rs
        ├── error.rs
        ├── arrow.rs   # feature-gated: "arrow"
        └── io/
            ├── mod.rs
            └── parquet.rs
```

## Build Instructions

```bash
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
cargo build --manifest-path rust/Cargo.toml
```

To build and run tests:

```bash
cargo test --manifest-path rust/Cargo.toml
```

## Usage Example

```rust
use cudf::{Resources, BinaryOp, DataType};
use cudf::io::parquet::read_parquet;
use cudf_sys::cudfTypeId_t;
use std::path::Path;

fn main() -> Result<(), cudf::CudfError> {
    // Create a resources handle. This allocates a CUDA stream.
    let resources = Resources::new()?;

    // Read a Parquet file into a GPU table.
    let table = read_parquet(Path::new("/path/to/data.parquet"), &resources)?;

    let num_rows = table.num_rows()?;
    let num_cols = table.num_columns()?;
    println!("Loaded {} rows x {} columns", num_rows, num_cols);

    // Get borrowed column views (valid while `table` lives).
    let col0 = table.column(0)?;
    let col1 = table.column(1)?;

    // Element-wise addition, producing a new owning column.
    let result = col0.binary_op(
        &col1,
        BinaryOp::Add,
        DataType::new(cudfTypeId_t::CUDF_TYPE_INT32),
        &resources,
    )?;

    println!(
        "Sum column: {} elements, {} nulls",
        result.size()?,
        result.null_count()?,
    );

    // Write the source table back to disk.
    table.write_parquet(Path::new("/tmp/out.parquet"), &resources)?;

    // Handles drop here, automatically freeing GPU memory.
    Ok(())
}
```

## Feature Flags

| Flag | Default | Effect |
|------|---------|--------|
| `doc-only` | off | Skips native library discovery. Use for `cargo doc` without a GPU environment. |
| `arrow` | off | Enables Arrow RecordBatch interop via the Arrow C Data Interface. Pulls in `arrow = "54"` with the `ffi` feature. |

Enable features with:

```bash
cargo build --features arrow --manifest-path rust/Cargo.toml
```

## Error Handling

All fallible methods return `Result<T, CudfError>`. The `CudfError` enum has
two variants:

```rust
pub enum CudfError {
    Error(String),   // error text from cudfGetLastErrorText()
    Unknown,         // C API returned failure but no error text was set
}
```

The internal `check_cudf()` helper converts a `cudfError_t` return value into a
`Result`, capturing the thread-local error string on failure. Because the error
string is thread-local in the C API, always retrieve the error before calling
any other C API function on the same thread.

## Thread Safety

`Resources`, `Column`, and `Table` all implement `Send`. They can be moved to
another thread. They do not implement `Sync`. Do not share a reference to the
same handle across threads without external synchronization (e.g. a `Mutex`).

Each thread should own its own `Resources` handle with its own CUDA stream to
avoid stream ordering issues.

## Regenerating FFI Bindings

The `cudf-sys` bindings in `rust/cudf-sys/src/bindings.rs` are pre-generated
and checked into the repository. You only need to regenerate them if the C
headers change.

Requirements: `bindgen` CLI and `libclang`.

```bash
bash rust/scripts/generate-bindings.sh
```

To verify that the checked-in bindings match the current headers without
overwriting them:

```bash
bash rust/scripts/generate-bindings.sh --check
```

## Supported Operations

| Module | Types / Functions |
|--------|-------------------|
| `resources` | `Resources::new()`, `set_stream()` |
| `column` | `Column::size()`, `null_count()`, `data_type()` |
| `table` | `Table::num_rows()`, `num_columns()`, `column(i)` |
| `io::parquet` | `read_parquet()`, `Table::write_parquet()` |
| `binaryop` | `Column::binary_op()`, `BinaryOp` enum, `DataType` |
| `join` | `Table::inner_join()` |
| `arrow` | `Table::from_arrow()`, `Table::to_arrow()` (feature: `arrow`) |

## Data Types

Type IDs come from `cudf_sys::cudfTypeId_t`. Common values:

```rust
cudfTypeId_t::CUDF_TYPE_INT8
cudfTypeId_t::CUDF_TYPE_INT32
cudfTypeId_t::CUDF_TYPE_INT64
cudfTypeId_t::CUDF_TYPE_FLOAT32
cudfTypeId_t::CUDF_TYPE_FLOAT64
cudfTypeId_t::CUDF_TYPE_STRING
cudfTypeId_t::CUDF_TYPE_BOOL8
cudfTypeId_t::CUDF_TYPE_TIMESTAMP_MILLISECONDS
```

For decimal types, use `DataType::with_scale(id, scale)` to specify the fixed
decimal scale.
