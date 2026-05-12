# Security Policy

cuDF is a GPU-accelerated DataFrame library (libcudf C++/CUDA, pylibcudf Cython
bindings, cudf and cudf-polars Python packages, dask-cudf, cudf_kafka, and Java
bindings consumed by Spark RAPIDS). Because it is a library — not a service —
its security posture is shaped primarily by the file-format parsers,
compression codecs, and bindings that sit at its trust boundary with caller-
supplied data.

This document covers what to report, how to report it, the threats we consider
in scope, and the assumptions callers are expected to satisfy.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected component (e.g. `libcudf` ORC reader, `cudf_kafka`, Java JNI bindings)
- cuDF / libcudf version, CUDA version, GPU model, and OS
- Reproduction steps and a minimal proof-of-concept (PoC) input
- Impact assessment (memory corruption, info disclosure, DoS, etc.)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix development,
and coordinated disclosure. More on NVIDIA's response process:
<https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library (C++/CUDA core with Python, Cython, and Java
bindings).

**Primary security responsibility:** Safely parse, transform, and emit tabular
data on the GPU. cuDF is invoked in-process by trusted callers (a Python
interpreter, a Dask worker, a Spark RAPIDS executor, a custom C++/Java
application) and inherits the caller's privilege.

**Components and trust boundaries:**

- **libcudf** (`cpp/`) — C++/CUDA core. The file-format I/O subsystem
  (`cpp/src/io/{parquet,orc,json,csv,avro,text}`) and its compression codecs
  (`cpp/src/io/comp/` — brotli, snappy, gzip/zlib, bz2 implemented in-house)
  are the primary attack surface: they parse untrusted bytes into GPU memory.
- **pylibcudf / cudf** (`python/pylibcudf`, `python/cudf`) — Python bindings.
  Public readers (`cudf.read_parquet`, `read_orc`, `read_json`, `read_csv`,
  `read_avro`, …) pass file paths, URLs, or buffers into libcudf.
- **cudf.pandas** — zero-code-change pandas accelerator that proxies pandas
  APIs onto cuDF; preserves caller's trust assumptions about inputs.
- **cudf-polars** (`python/cudf_polars`) — Polars GPU execution engine.
- **dask-cudf / custreamz** — Dask integration; participates in cluster
  serialization (pickle) when distributed.
- **cudf_kafka** (`cpp/libcudf_kafka`, `python/cudf_kafka`) — Kafka consumer
  for streaming ingest; relies on librdkafka for transport security.
- **Java bindings** (`java/`, JNI in `java/src/main/native/src/`) — consumed
  by external runtimes such as Spark RAPIDS. JNI entry points trust the JVM
  caller to supply well-formed `ColumnView` / `Table` references.
- **Remote I/O** is delegated to [KvikIO](https://github.com/rapidsai/kvikio)
  (selected by URL scheme regex in `cpp/src/io/utilities/datasource.cpp`) and
  to fsspec on the Python side (S3, GCS, HTTP, WebHDFS).

**Out of scope for this policy:** vulnerabilities in CUDA, the NVIDIA driver,
KvikIO, librdkafka, pyarrow, pandas, cupy, Dask, RMM, or the JVM. Report
those to their respective projects (NVIDIA driver and CUDA bugs still go to
PSIRT).

## Threat Model

The threats below trace to specific components in this repository. Several
have already been observed and fixed or are being remediated through the
[RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207); they
are listed here so that callers and integrators understand the
classes of bugs the library defends against.

1. **Malformed file-format inputs causing memory corruption.**
   The ORC, Parquet, JSON, CSV, and Avro readers in `cpp/src/io/` parse
   attacker-controlled bytes into GPU buffers. Historical findings include
   heap buffer overflow, integer underflow, and OOB reads in the ORC reader;
   heap overflow and CUDA-kernel OOB in the CSV reader; and multiple memory
   safety bugs in the JSON tokenizer. A hostile Parquet/ORC/JSON/CSV/Avro
   file is the canonical exploit vector.

2. **Decompression bombs (DoS).**
   The in-house gzip/zlib, snappy, brotli, and bz2 decoders in
   `cpp/src/io/comp/` do not enforce an output-size ceiling. A small
   compressed payload embedded in a CSV or JSON input can expand to
   exhaust GPU or host memory.

3. **Integer underflow on truncated / hostile metadata.**
   Footer and stripe-metadata reads (e.g. Parquet footer length, ORC stripe
   sizes) can underflow when the file is truncated or claims sizes
   inconsistent with its actual contents, leading to oversized allocations
   or OOB indexing further down the pipeline.

4. **Unsafe deserialization on distributed paths.**
   `dask-cudf` and `cudf_kafka` participate in pickle-based serialization
   when used with Dask Distributed. Pickle is unsafe against untrusted
   peers; treat the Dask cluster network and Kafka topic as a trust
   boundary.

5. **Code-construction-from-parsed-input.**
   The PTX JIT path historically constructed inline assembly via string
   concatenation from values that traced back to parsed inputs. Any future
   feature that templates kernels, PTX, or SQL fragments from caller data
   inherits the same risk class.

6. **JNI / FFI boundary mistrust.**
   The Java bindings (`java/src/main/native/src/*Jni.cpp`) accept native
   pointers and sizes from the JVM. A buggy or malicious caller can pass
   mismatched lengths, freed pointers, or wrong types; libcudf cannot
   distinguish these from valid calls.

7. **Environment-variable-controlled I/O rerouting.**
   `LIBCUDF_IO_REROUTE_LOCAL_DIR_PATTERN` / `LIBCUDF_IO_REROUTE_REMOTE_DIR_PATTERN`
   (see `cpp/src/io/utilities/datasource.cpp`) and `LIBCUDF_MMAP_ENABLED`
   change I/O behavior at runtime. A process whose environment is influenced
   by an attacker can be redirected to fetch from an attacker-controlled
   remote URL in place of a local file.

## Critical Security Assumptions

cuDF is a library and inherits the caller's privilege; the following are
assumed of the caller / deployer.

- **Inputs may be hostile, but the caller decides whether to trust them.**
  cuDF endeavors to fail safely on malformed file-format inputs, but
  callers parsing data from external sources should run cuDF in a process
  with the smallest viable blast radius (separate process, container,
  resource limits).

- **The host process, environment, and library load path are trusted.**
  cuDF does not authenticate environment variables (`LIBCUDF_*`), dynamic
  loaders, or sibling libraries. A caller who lets an attacker influence
  these is exposed to the rerouting and configuration threats above.

- **Transport security is provided externally.**
  cuDF does not implement TLS, authentication, or authorization. Remote
  reads via KvikIO/fsspec depend on those libraries (and the underlying
  S3/HTTP client and OS trust store) for confidentiality and integrity.
  Kafka transport security is the responsibility of librdkafka
  configuration provided by the caller.

- **Distributed cluster peers are mutually trusted.**
  Dask-cuDF and cudf_kafka pickle-based serialization is unsafe across
  trust boundaries. Run distributed clusters on private networks with
  authenticated peers; do not accept pickled cuDF payloads from untrusted
  sources.

- **JNI and FFI callers preserve API contracts.**
  Java/Spark and other native callers are expected to pass valid
  pointers, live `ColumnView`s, and accurate sizes. libcudf performs
  internal validation where practical but cannot fully defend against a
  hostile in-process caller.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU, or co-tenants on a shared host, can
  potentially observe each other's GPU memory through driver-level
  side channels. cuDF assumes the caller has provisioned the GPU
  appropriately (MIG, exclusive process, container isolation) when
  confidentiality matters.

- **Decompression ratios are bounded by the caller's resource limits.**
  Until output-size ceilings are enforced inside the codecs, callers
  ingesting untrusted compressed input should impose memory and time
  limits on the cuDF process (cgroups, ulimit, container quotas).

## Supported Versions

Security fixes are issued against the current release line published on the
[RAPIDS release schedule](https://docs.rapids.ai/releases/). Older minor
releases are generally not backported; upgrade to the latest supported
version to receive fixes.

## Dependency Security

cuDF tracks CVEs in its direct dependencies (notably `pyarrow`, `pandas`,
`numpy`, `numba`, `cuda-python`, `cupy`, `kvikio`, `librdkafka`, and — on
the Java side — `logback`, `plexus-utils`, and the Go toolchain used by
build tooling). Dependency updates ship with regular releases; high-severity
upstream CVEs may trigger out-of-band patch releases.
