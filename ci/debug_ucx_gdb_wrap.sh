#!/bin/bash
# TEMPORARY (UCX segfault debugging, not for merge).
# Wrapper that runs a command under gdb in batch mode and dumps as much context
# as possible if the process receives SIGSEGV. libcuda.so is stripped, so frames
# inside it show up as raw offsets; the register dump, disassembly and build-id
# printed here let NVIDIA symbolize them offline.
if ! command -v gdb > /dev/null; then
    exec "$@"
fi
out="$(mktemp)"
gdb -q -batch \
    -ex "set pagination off" \
    -ex "set print thread-events off" \
    -ex "handle SIGSEGV stop print nopass" \
    -ex "run" \
    -ex "echo \n===== GDB: stopped =====\n" \
    -ex "bt" \
    -ex "echo \n===== GDB: info registers =====\n" \
    -ex "info registers" \
    -ex "echo \n===== GDB: disassembly around pc =====\n" \
    -ex "x/24i \$pc-48" \
    -ex "info symbol \$pc" \
    -ex "echo \n===== GDB: shared libraries =====\n" \
    -ex "info sharedlibrary" \
    -ex "echo \n===== GDB: all thread backtraces =====\n" \
    -ex "thread apply all bt" \
    -ex "echo \n===== GDB: mappings =====\n" \
    -ex "info proc mappings" \
    --args "$@" 2>&1 | tee "${out}"

# gdb -batch exits 0 even if the inferior crashed; recover the real status so
# mpirun / the caller see the failure.
if grep -q "Program received signal\|Program terminated with signal" "${out}"; then
    exit 139
fi
code="$(sed -n 's/.*exited with code \([0-9]*\)\].*/\1/p' "${out}" | tail -n 1)"
exit $(( 10#${code:-0} ))
