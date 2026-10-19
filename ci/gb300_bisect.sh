#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUDA_MAJOR="${RAPIDS_PY_CUDA_SUFFIX#cu}"

rapids-generate-pip-constraints py_test_cudf "${PIP_CONSTRAINT}" constraints

# cudf wheels from the main build of 816f682244, which segfaults on GB300.
CUDF_SHA=816f682244c080b438792eec6bc8eda6c3b0583a
CUDF_RUN_ID=39100160924

WORK="$(mktemp -d)"
mkdir -p "${WORK}/wheels" "${WORK}/src"
gh run download "${CUDF_RUN_ID}" --repo NVIDIA/cudf \
    --name "cudf_wheel_cpp_libcudf_aarch64_cu${CUDA_MAJOR}" \
    --name "cudf_wheel_python_pylibcudf_aarch64_abi3_cu${CUDA_MAJOR}" \
    --dir "${WORK}/wheels"
git archive "${CUDF_SHA}" python/pylibcudf/tests | tar -x -C "${WORK}/src"

python -m venv "${WORK}/env"
# shellcheck disable=SC1091
. "${WORK}/env/bin/activate"
rapids-pip-retry install \
    --prefer-binary \
    --constraint "${PIP_CONSTRAINT}" \
    "$(find "${WORK}/wheels" -name "libcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")" \
    "$(find "${WORK}/wheels" -name "pylibcudf_${RAPIDS_PY_CUDA_SUFFIX}*.whl")[test]"
python -m pip list 2>/dev/null | grep -i -E "rmm|kvikio|nvcomp|cuda|cudf"

pushd "${WORK}/src/python/pylibcudf/tests" > /dev/null
SUMMARY=()

apt-get update -qq && apt-get install -y -qq --no-install-recommends gdb > /dev/null

# xdist worker interpreter that runs under gdb, with gdb output kept off execnet's stdio channel.
REAL_PYTHON="$(command -v python)"
GDB_PYTHON="${WORK}/gdbpython"
cat > "${GDB_PYTHON}" <<WRAPPER
#!/bin/bash
exec gdb -q -batch -nx \\
  -ex "set logging file ${WORK}/gdb-\$\$.log" \\
  -ex "set logging overwrite on" \\
  -ex "set logging redirect on" \\
  -ex "set logging enabled on" \\
  -ex "set print thread-events off" \\
  -ex "set print inferior-events off" \\
  -ex "set pagination off" \\
  -ex "handle SIGSEGV stop print nopass" \\
  -ex "handle SIGPIPE nostop noprint pass" \\
  -ex run \\
  -ex "bt 60" \\
  -ex "p \\\$_siginfo._sifields._sigfault.si_addr" \\
  -ex "info registers" \\
  -ex "x/12i \\\$pc-24" \\
  -ex "info proc mappings" \\
  -ex "info threads" \\
  -ex "thread apply all bt 40" \\
  -ex "info sharedlibrary" \\
  --args "${REAL_PYTHON}" "\$@"
WRAPPER
chmod +x "${GDB_PYTHON}"

for attempt in 1 2 3 4; do
  rm -f "${WORK}"/gdb-*.log
  set +e
  timeout 25m python -m pytest -p no:cacheprovider --tx "8*popen//python=${GDB_PYTHON}" \
    --dist=worksteal -q . > "${WORK}/xdist-${attempt}.log" 2>&1
  status=$?
  set -e
  crash=$(grep -o "worker '[^']*' crashed while running '[^']*'" "${WORK}/xdist-${attempt}.log" | head -n 1 || true)
  SUMMARY+=("attempt=${attempt} exit=${status} ${crash}")
  tail -n 3 "${WORK}/xdist-${attempt}.log"
  segv_log=$(grep -l "received signal SIGSEGV" "${WORK}"/gdb-*.log 2>/dev/null | head -n 1 || true)
  if [[ -n "${segv_log}" ]]; then
    rapids-logger "Attempt ${attempt}: all-thread backtraces from ${segv_log}"
    sed -n '/received signal SIGSEGV/,/^Start Addr/p' "${segv_log}" | grep -v "^Start Addr" || true
    addr=$(grep -oE '^\$1 = \(void \*\) 0x[0-9a-f]+' "${segv_log}" | grep -oE '0x[0-9a-f]+' || true)
    rapids-logger "Mapping containing fault address ${addr:-unknown}"
    if [[ -n "${addr}" ]]; then
      python - "${addr}" "${segv_log}" <<'PY'
import sys

addr = int(sys.argv[1], 16)
found = False
for line in open(sys.argv[2], errors="replace"):
    fields = line.split()
    if len(fields) >= 2 and fields[0].startswith("0x") and fields[1].startswith("0x"):
        try:
            start, end = int(fields[0], 16), int(fields[1], 16)
        except ValueError:
            continue
        if start <= addr < end:
            print(line.rstrip())
            found = True
if not found:
    print("fault address is not mapped")
PY
    fi
    rapids-logger "Other threads"
    sed -n '/^  Id   Target Id/,$p' "${segv_log}" \
      | grep -v -E "^0x[0-9a-f]+ +0x[0-9a-f]+ +(Yes|No) " | grep -v -E "^#[0-9]+ .*(pthread_cond_wait|libc\.so\.6|libstdc\+\+|condition_variable)" \
      | head -n 300 || true
    break
  fi
done

popd > /dev/null

rapids-logger "GB300 xdist gdb summary"
printf '%s\n' "${SUMMARY[@]}"
