#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# run_examples.sh — generates sample Parquet data, runs both hybrid-scan
# examples, then cleans up.  Safe to invoke from any working directory;
# the script cds into its own directory before running mvn.
#
# Behaviour:
#   * Prechecks that the parent ai.rapids:cudf jar (with a CUDA classifier)
#     is installed in the local Maven repository (~/.m2/repository) and
#     fails fast with a clear remediation message if it isn't.
#   * Builds the example module on first run (no compiled *.class files
#     under target/classes/ai/rapids/cudf/examples/). Pass -b/--build to
#     force a rebuild even when classes are present.
#   * Runs the three example stages: data generation, two-step IO, chunked
#     pipeline.

set -e

usage() {
    cat <<'USAGE'
run_examples.sh -- runs all hybrid-scan Java examples end-to-end

USAGE:
  run_examples.sh [OPTIONS]

OPTIONS:
  -h, --help    Show this help and exit.
  -b, --build   Force `mvn package -DskipTests` even when target/classes
                already contains compiled .class files. Without this flag,
                the script auto-builds only when target/classes is empty.

PRECHECK:
  The script verifies that ai.rapids:cudf:<version>-<cuda.classifier> is
  installed in ~/.m2/repository before doing anything else; if not, it
  prints the expected jar path and the `mvn install -DskipTests` command
  to install it.

STAGES:
  0. Build the example module (only when needed).
  1. Generate a deterministic sample Parquet file.
  2. HybridScanIoExample (legacy reader vs hybrid-scan two-step).
  3. HybridScanPipelineExample (chunked all-columns pipeline).

  Stage 0 is skipped when target/classes already has *.class files and
  -b/--build was not passed.
USAGE
}

FORCE_BUILD=0
while (( $# > 0 )); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -b|--build)
            FORCE_BUILD=1
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo >&2
            usage >&2
            exit 64
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Anchor every subsequent `mvn` call to the example module's pom regardless
# of where the user invoked us from. Without this, `mvn exec:java` in stages
# 1-3 would inherit the caller's cwd and fail with "no POM in this directory".
cd "${SCRIPT_DIR}"

DATA_FILE="${SCRIPT_DIR}/test_data.parquet"
MVN_FLAGS="-q"   # remove -q to see full Maven output

# ── helpers ────────────────────────────────────────────────────────────────
banner() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "═══════════════════════════════════════════════════════════════════"
}

step() { echo "▶  $*"; }

# ── cleanup ────────────────────────────────────────────────────────────────
cleanup() {
    if [[ -f "${DATA_FILE}" ]]; then
        step "Cleaning up: removing ${DATA_FILE}"
        rm -f "${DATA_FILE}"
    fi
}
trap cleanup EXIT

# ── preflight: ai.rapids:cudf jar must be present in the local Maven repo ──
# Field-split each pom line on '<' and '>'; the first <version> tag in the
# pom is the project's <version> (Maven puts it in the identification block
# at the top of the file). <modelVersion> doesn't match the /<version>/
# regex because its '<' is followed by 'm', not 'v'.
POM="${SCRIPT_DIR}/pom.xml"
proj_version=$(awk    -F'[<>]' '/<version>/        {print $3; exit}' "${POM}")
cuda_classifier=$(awk -F'[<>]' '/<cuda.classifier>/{print $3; exit}' "${POM}")
cuda_classifier="${cuda_classifier:-cuda12}"

mvn_repo="${HOME}/.m2/repository"
expected_jar="${mvn_repo}/ai/rapids/cudf/${proj_version}/cudf-${proj_version}-${cuda_classifier}.jar"

if [[ ! -f "${expected_jar}" ]]; then
    cat >&2 <<EOF
ERROR: ai.rapids:cudf:${proj_version} (classifier ${cuda_classifier}) is not
       installed in your local Maven repository.

       Expected jar:
         ${expected_jar}

       Searched local Maven repo:
         ${mvn_repo}

       To install it, build the parent cuDF Java module from the cuDF tree:
         cd <cudf-repo>/java && mvn install -DskipTests

       If your libcudf was built for a different CUDA major version, set the
       classifier when you run that build, e.g.:
         cd <cudf-repo>/java && mvn install -DskipTests -Dcuda.classifier=cuda11
       and then re-run this script.
EOF
    exit 2
fi
step "Found ${expected_jar#${mvn_repo}/}"

# ── stage 0: build (optional) ─────────────────────────────────────────────
CLASSES_DIR="${SCRIPT_DIR}/target/classes/ai/rapids/cudf/examples"
NEED_BUILD=0
# `compgen -G` is a bash builtin that exits 0 iff at least one path matches
# the glob — no nullglob toggle, no array, no `ls 2>/dev/null` dance.
if [[ "${FORCE_BUILD}" -eq 1 ]]; then
    step "Build forced via -b/--build"
    NEED_BUILD=1
elif ! compgen -G "${CLASSES_DIR}/*.class" > /dev/null; then
    step "No *.class files under ${CLASSES_DIR#${SCRIPT_DIR}/} -- building"
    NEED_BUILD=1
else
    step "Reusing existing *.class files (pass -b to force rebuild)"
fi
if [[ "${NEED_BUILD}" -eq 1 ]]; then
    banner "Stage 0 — mvn package -DskipTests"
    ( cd "${SCRIPT_DIR}" && mvn ${MVN_FLAGS} package -DskipTests )
    echo "✔  Build complete."
fi

# ── stage 1: generate sample data ─────────────────────────────────────────
banner "Stage 1 — Generate sample Parquet file"
step "Output: ${DATA_FILE}"
step "Schema: id INT, zip_code INT, num_units INT  |  5 row groups x 50,000 rows"
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.examples.GenerateSampleParquetFileMain \
    -Dexec.args="${DATA_FILE}"
echo "✔  Sample data written."

# ── stage 2: HybridScanIoExample ──────────────────────────────────────────
banner "Stage 2 — HybridScanIoExample (legacy vs hybrid-scan two-step)"
step "Filter: zip_code > 145000"
step "Compares the legacy Table.readParquet path against the two-step hybrid scan."
step "Includes a [Hybrid: PageIndex Filtering] variant that prunes pages via the page index."
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanIoExample \
    -Dexec.args="${DATA_FILE} zip_code 145000"
echo "✔  HybridScanIoExample complete."

# ── stage 3: HybridScanPipelineExample ────────────────────────────────────
banner "Stage 3 — HybridScanPipelineExample (chunked pipeline, no filter)"
step "Row group batch size: 0 (no limit — one pass for all row groups)"
step "Chunk size:           0 (no limit — one cuDF Table chunk per pass)"
step "Demonstrates the streaming/chunked all-columns API."
mvn ${MVN_FLAGS} exec:java \
    -Dexec.mainClass=ai.rapids.cudf.examples.HybridScanPipelineExample \
    -Dexec.args="${DATA_FILE} 0 0"
echo "✔  HybridScanPipelineExample complete."

# ── done ──────────────────────────────────────────────────────────────────
banner "All examples finished successfully"
# cleanup() runs automatically via trap EXIT
