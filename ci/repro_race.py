# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import glob
import os
import subprocess
import sys
import threading
import time

KINDS = ["cuMemAlloc", "cuMemAllocAsync", "cuMemHostAlloc", "vmm"]
SECS = float(os.environ.get("REPRO_SECS", "15"))
CHURN_THREADS = 2
QUERY_THREADS = 2


def child(kind):
    cu = ctypes.CDLL("libcuda.so.1")
    assert cu.cuInit(0) == 0
    dev = ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    cu.cuCtxSetCurrent(ctx)
    page = os.sysconf("SC_PAGE_SIZE")
    stop = threading.Event()
    counts = {"churn": 0, "query_ok": 0, "query_err": 0}
    lock = threading.Lock()

    persistent = ctypes.c_uint64()
    assert (
        cu.cuMemAlloc_v2(ctypes.byref(persistent), ctypes.c_size_t(1 << 20))
        == 0
    )

    def query(ptr, n):
        start = ptr & ~(page - 1)
        end = (ptr + n + page - 1) & ~(page - 1)
        fd = ctypes.c_int(-1)
        res = cu.cuMemGetHandleForAddressRange(
            ctypes.byref(fd),
            ctypes.c_uint64(start),
            ctypes.c_size_t(end - start),
            1,
            ctypes.c_ulonglong(0),
        )
        if res == 0:
            os.close(fd.value)
        return res

    def churn():
        cu.cuCtxSetCurrent(ctx)
        stream = ctypes.c_void_p()
        cu.cuStreamCreate(ctypes.byref(stream), 0)
        n = 0
        while not stop.is_set():
            size = 1 << (12 + n % 12)
            p = ctypes.c_uint64()
            hp = ctypes.c_void_p()
            if kind == "cuMemAlloc":
                cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(size))
                cu.cuMemFree_v2(p)
            elif kind == "cuMemAllocAsync":
                cu.cuMemAllocAsync(
                    ctypes.byref(p), ctypes.c_size_t(size), stream
                )
                cu.cuMemFreeAsync(p, stream)
                if n % 64 == 0:
                    cu.cuStreamSynchronize(stream)
            elif kind == "cuMemHostAlloc":
                cu.cuMemHostAlloc(ctypes.byref(hp), ctypes.c_size_t(size), 0)
                cu.cuMemFreeHost(hp)
            else:
                prop = (ctypes.c_char * 64)()
                ints = ctypes.cast(prop, ctypes.POINTER(ctypes.c_int))
                ints[0] = 1
                ints[2] = 1
                ints[3] = dev.value
                h = ctypes.c_uint64()
                gran = 2 << 20
                if (
                    cu.cuMemCreate(
                        ctypes.byref(h),
                        ctypes.c_size_t(gran),
                        prop,
                        ctypes.c_ulonglong(0),
                    )
                    == 0
                ):
                    va = ctypes.c_uint64()
                    cu.cuMemAddressReserve(
                        ctypes.byref(va),
                        ctypes.c_size_t(gran),
                        ctypes.c_size_t(0),
                        ctypes.c_uint64(0),
                        ctypes.c_ulonglong(0),
                    )
                    cu.cuMemMap(
                        va,
                        ctypes.c_size_t(gran),
                        ctypes.c_size_t(0),
                        h,
                        ctypes.c_ulonglong(0),
                    )
                    cu.cuMemUnmap(va, ctypes.c_size_t(gran))
                    cu.cuMemAddressFree(va, ctypes.c_size_t(gran))
                    cu.cuMemRelease(h)
            n += 1
            with lock:
                counts["churn"] += 1

    def querier():
        cu.cuCtxSetCurrent(ctx)
        while not stop.is_set():
            res = query(persistent.value, 1 << 20)
            with lock:
                counts["query_ok" if res == 0 else "query_err"] += 1

    threads = [threading.Thread(target=churn) for _ in range(CHURN_THREADS)]
    threads += [threading.Thread(target=querier) for _ in range(QUERY_THREADS)]
    for t in threads:
        t.start()
    time.sleep(SECS)
    stop.set()
    for t in threads:
        t.join()
    print("race", kind, counts, flush=True)


def main():
    for variant in os.environ.get("REPRO_LIBCUDA", "default,host").split(","):
        for kind in KINDS:
            env = dict(os.environ)
            if variant == "host":
                hosts = [
                    p
                    for p in glob.glob("/usr/lib/*/libcuda.so.*")
                    if "compat" not in p
                ]
                if not hosts:
                    continue
                env["LD_PRELOAD"] = os.path.realpath(sorted(hosts)[0])
            r = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    os.path.abspath(__file__),
                    "child",
                    kind,
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=SECS + 120,
            )
            tag = (
                f"signal {-r.returncode}"
                if r.returncode < 0
                else f"rc={r.returncode}"
            )
            print(
                f"[{variant}] race {kind:16s} {tag}  {r.stdout.strip()[-160:]}",
                flush=True,
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        main()
