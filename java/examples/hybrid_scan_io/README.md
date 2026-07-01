# cudf Java hybrid_scan_io examples

Java examples demonstrating how to use the experimental
`ai.rapids.cudf.HybridScanReader` (the Java binding for
`cudf::io::parquet::experimental::hybrid_scan_reader`, marked `@Experimental` on the Java
side) to read Parquet files subject to selective filter expressions.

## Examples

| Class | Purpose |
|---|---|
| `GenerateSampleParquetFileMain` | Generates a deterministic five-row-group (250,000-row) Parquet file with COLUMN-level statistics (page index). Run this first. |
| `HybridScanIoExample` | Reads a file three ways: (1) legacy `Table.readParquet`, (2) hybrid two-step with `ALL_TRUE` row mask (stats-only pruning), and (3) hybrid two-step with `PAGE_INDEX_STATS` row mask (page-index pruning). Prints row counts and elapsed time for each scenario. |
| `HybridScanPipelineExample` | Splits the file into row-group passes bounded by a byte limit, then streams each pass as cuDF Table chunks via the chunked all-columns API. Demonstrates bounded GPU memory usage with no filter applied. |

`Util.java` contains shared helpers: reading a Parquet file or just its footer into a
`HostMemoryBuffer`, reading a single targeted byte range from a file, and copying byte
ranges to device buffers (from either a host buffer or directly from a file).

## Building

The examples expect that the parent `ai.rapids:cudf` artifact has been built and installed
locally. From the repo root:

```bash
cd java
mvn -DskipTests install   # installs cudf-java jar into ~/.m2 so examples can depend on it
cd examples/hybrid_scan_io
mvn package               # compiles and packages only this example module
```

## Generating sample data

The examples read from a Parquet file with three integer columns (`id`, `zip_code`,
`num_units`) split into five row groups of 50,000 rows each, written with COLUMN-level
statistics so that a page index is available for page-level pruning. Generate it before
running the examples:

```bash
# Generate the sample Parquet file used by the examples
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.GenerateSampleParquetFileMain \
  -Dexec.args="/tmp/sample.parquet"
```

The `zip_code` values are deliberately distributed across five non-overlapping bases so
that a filter threshold of 145,000 prunes:

- 1 row group entirely via row-group statistics
- 2 row groups partially at the page level (2 pages in one, 1 page in the other)
- 2 row groups not at all (all rows survive)

## Running

The fastest way to exercise all three examples is the bundled `run_examples.sh`:

```bash
./run_examples.sh                # data-gen + io + pipeline
./run_examples.sh -b             # also force `mvn package -DskipTests` first
./run_examples.sh --help         # full usage
```

`run_examples.sh`:

- Prechecks that the `ai.rapids:cudf` jar is installed in `~/.m2/repository`
  and prints a clear error (with the `mvn install -DskipTests` invocation)
  if it isn't.
- Auto-builds the example module on first run when
  `target/classes/ai/rapids/cudf/examples/` has no `*.class` files.
- Accepts `-b` / `--build` to force a rebuild even when classes are already
  present.

To run a single example by hand (after `mvn package` has been run on the
examples module):

```bash
# Single/two-step example
# Arguments:
#   parquet-file   Path to a Parquet file (use GenerateSampleParquetFileMain to create one)
#   column-name    Name of an integer column to filter on (e.g. zip_code)
#   int-literal    Integer threshold; rows where column-name > int-literal are kept
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanIoExample \
  -Dexec.args="/tmp/sample.parquet zip_code 145000"

# Chunked pipeline example
# Arguments:
#   parquet-file          Path to a Parquet file (use GenerateSampleParquetFileMain to create one)
#   row-group-batch-bytes Row group batch size in bytes: maximum total uncompressed size of the
#                         row groups in a single pass (batch). A pass is a batch of row groups
#                         whose combined uncompressed size fits within this limit.
#                         0 = no limit (all row groups in one pass).
#   chunk-bytes           Maximum size in bytes of a single output cuDF Table chunk within a pass.
#                         Controls how much decoded GPU memory is used at once per chunk.
#                         0 = no limit (entire pass materialised as one Table).
mvn exec:java \
  -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanPipelineExample \
  -Dexec.args="/tmp/sample.parquet 67108864 16777216"
```

## Notes on filters

Because the Parquet schema may not be known to the application at AST construction time,
filter expressions for the hybrid scan reader use
`ai.rapids.cudf.ast.ColumnNameReference` to refer to columns by name (see the example
source). The reader resolves the names against the file schema during materialization.
