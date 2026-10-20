# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import glob
import os
import subprocess
import sys
import threading

CASES = [
    "cuMemAlloc_1MiB",
    "cuMemAlloc_256B",
    "cuMemAllocAsync_1MiB",
    "cuMemAllocAsync_256B",
    "vmm_2MiB",
    "managed_1MiB",
    "pinned_host_1MiB",
    "host_registered_1MiB",
    "hostpool_1MiB",
    "vmm_host_2MiB",
]
MODES = ["driver", "ucx", "ucx_thread"]


def libcuda_maps():
    with open("/proc/self/maps") as f:
        return sorted({ln.split()[-1] for ln in f if "libcuda.so" in ln})


def ck(res, what):
    if res != 0:
        print(f"{what} failed: {res}", flush=True)
        sys.exit(3)


def child(mode, case, iters):
    cu = ctypes.CDLL("libcuda.so.1")
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain")
    ck(cu.cuCtxSetCurrent(ctx), "setcurrent")
    print("libcuda:", libcuda_maps(), flush=True)

    stream = ctypes.c_void_p()
    ck(cu.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")

    size = (
        256
        if case.endswith("256B")
        else (2 << 20 if "2MiB" in case else 1 << 20)
    )

    page = os.sysconf("SC_PAGE_SIZE")
    keep = []
    pool_holder = []
    ptr_v = ctypes.c_void_p()

    def alloc():
        ptr = ctypes.c_uint64()
        if case.startswith("cuMemAllocAsync"):
            ck(
                cu.cuMemAllocAsync(
                    ctypes.byref(ptr), ctypes.c_size_t(size), stream
                ),
                "allocAsync",
            )
            ck(cu.cuStreamSynchronize(stream), "sync")
            return ptr.value, lambda: (
                cu.cuMemFreeAsync(ctypes.c_uint64(ptr.value), stream),
                cu.cuStreamSynchronize(stream),
            )
        if case.startswith("cuMemAlloc_"):
            ck(
                cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(size)),
                "cuMemAlloc",
            )
            return ptr.value, lambda: cu.cuMemFree_v2(
                ctypes.c_uint64(ptr.value)
            )
        if case.startswith("managed"):
            ck(
                cu.cuMemAllocManaged(
                    ctypes.byref(ptr), ctypes.c_size_t(size), 1
                ),
                "managed",
            )
            return ptr.value, lambda: cu.cuMemFree_v2(
                ctypes.c_uint64(ptr.value)
            )
        if case.startswith("pinned_host"):
            ck(
                cu.cuMemHostAlloc(
                    ctypes.byref(ptr_v), ctypes.c_size_t(size), 0
                ),
                "hostalloc",
            )
            return ptr_v.value, lambda: cu.cuMemFreeHost(ptr_v)
        if case.startswith("host_registered"):
            buf = ctypes.create_string_buffer(size + page)
            addr = (ctypes.addressof(buf) + page - 1) & ~(page - 1)
            ck(
                cu.cuMemHostRegister_v2(
                    ctypes.c_void_p(addr), ctypes.c_size_t(size), 0
                ),
                "hostregister",
            )
            keep.append(buf)
            return addr, lambda: cu.cuMemHostUnregister(ctypes.c_void_p(addr))
        if case.startswith("hostpool"):
            if not pool_holder:
                props = (ctypes.c_char * 256)()
                ctypes.cast(props, ctypes.POINTER(ctypes.c_int))[0] = 1
                ctypes.cast(props, ctypes.POINTER(ctypes.c_int))[2] = 2
                pool = ctypes.c_void_p()
                ck(
                    cu.cuMemPoolCreate(ctypes.byref(pool), props),
                    "cuMemPoolCreate(host)",
                )
                pool_holder.append(pool)
            ck(
                cu.cuMemAllocFromPoolAsync(
                    ctypes.byref(ptr),
                    ctypes.c_size_t(size),
                    pool_holder[0],
                    stream,
                ),
                "allocFromPool",
            )
            ck(cu.cuStreamSynchronize(stream), "sync")
            return ptr.value, lambda: (
                cu.cuMemFreeAsync(ctypes.c_uint64(ptr.value), stream),
                cu.cuStreamSynchronize(stream),
            )
        if case.startswith("vmm"):

            class Loc(ctypes.Structure):
                _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

            class Prop(ctypes.Structure):
                _fields_ = [
                    ("type", ctypes.c_int),
                    ("requestedHandleTypes", ctypes.c_int),
                    ("location", Loc),
                    ("win32HandleMetaData", ctypes.c_void_p),
                    ("allocFlags", ctypes.c_ubyte * 8),
                ]

            class Access(ctypes.Structure):
                _fields_ = [("location", Loc), ("flags", ctypes.c_int)]

            prop = Prop()
            prop.type = 1
            prop.location.type = 2 if "host" in case else 1
            prop.location.id = 0 if "host" in case else dev.value
            handle = ctypes.c_uint64()
            ck(
                cu.cuMemCreate(
                    ctypes.byref(handle),
                    ctypes.c_size_t(size),
                    ctypes.byref(prop),
                    ctypes.c_ulonglong(0),
                ),
                "cuMemCreate",
            )
            va = ctypes.c_uint64()
            ck(
                cu.cuMemAddressReserve(
                    ctypes.byref(va),
                    ctypes.c_size_t(size),
                    ctypes.c_size_t(0),
                    ctypes.c_uint64(0),
                    ctypes.c_ulonglong(0),
                ),
                "reserve",
            )
            ck(
                cu.cuMemMap(
                    ctypes.c_uint64(va.value),
                    ctypes.c_size_t(size),
                    ctypes.c_size_t(0),
                    handle,
                    ctypes.c_ulonglong(0),
                ),
                "map",
            )
            acc = Access()
            acc.location.type = 1
            acc.location.id = dev.value
            acc.flags = 3
            ck(
                cu.cuMemSetAccess(
                    ctypes.c_uint64(va.value),
                    ctypes.c_size_t(size),
                    ctypes.byref(acc),
                    ctypes.c_size_t(1),
                ),
                "setaccess",
            )
            return va.value, lambda: None
        raise SystemExit(f"bad case {case}")

    def dmabuf_query(ptr):
        base = ctypes.c_uint64()
        length = ctypes.c_size_t()
        cu.cuMemGetAddressRange_v2(
            ctypes.byref(base), ctypes.byref(length), ctypes.c_uint64(ptr)
        )
        b = base.value if base.value else ptr
        n = length.value if length.value else size
        start = b & ~(page - 1)
        end = (b + n + page - 1) & ~(page - 1)
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

    if mode == "driver":
        results = {}
        for i in range(iters):
            ptr, free = alloc()
            res = dmabuf_query(ptr)
            results[res] = results.get(res, 0) + 1
            free()
        print("driver results {err: count}:", results, flush=True)
        return

    if mode in ("ucx", "ucx_thread"):
        ucp = ctypes.CDLL("libucp.so.0")

        class Params(ctypes.Structure):
            _fields_ = [
                ("field_mask", ctypes.c_uint64),
                ("features", ctypes.c_uint64),
                ("request_size", ctypes.c_size_t),
                ("pad", ctypes.c_char * 512),
            ]

        class WParams(ctypes.Structure):
            _fields_ = [
                ("field_mask", ctypes.c_uint64),
                ("pad", ctypes.c_char * 512),
            ]

        class ReqParam(ctypes.Structure):
            _fields_ = [("pad", ctypes.c_char * 512)]

        maj, mnr = ctypes.c_uint(), ctypes.c_uint()
        ucp.ucp_get_version(
            ctypes.byref(maj), ctypes.byref(mnr), ctypes.byref(ctypes.c_uint())
        )
        print("ucp api", maj.value, mnr.value, flush=True)
        cfg = ctypes.c_void_p()
        ck(
            ucp.ucp_config_read(None, None, ctypes.byref(cfg)),
            "ucp_config_read",
        )
        p = Params()
        p.field_mask = 3
        p.features = 1
        p.request_size = 256
        context = ctypes.c_void_p()
        ck(
            ucp.ucp_init_version(
                maj, mnr, ctypes.byref(p), cfg, ctypes.byref(context)
            ),
            "ucp_init",
        )
        wp = WParams()
        worker = ctypes.c_void_p()
        ck(
            ucp.ucp_worker_create(
                context, ctypes.byref(wp), ctypes.byref(worker)
            ),
            "worker_create",
        )
        ucp.ucp_tag_recv_nbx.restype = ctypes.c_void_p
        ucp.ucp_tag_recv_nbx.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        results = {}
        bufs = [alloc() for _ in range(iters)]

        def post_all():
            for i, (ptr, _) in enumerate(bufs):
                req = ucp.ucp_tag_recv_nbx(
                    worker,
                    ptr,
                    size,
                    i + 1,
                    0xFFFFFFFFFFFFFFFF,
                    ctypes.byref(ReqParam()),
                )
                key = (
                    "err"
                    if (req is not None and req >= (1 << 64) - 100)
                    else "posted"
                )
                results[key] = results.get(key, 0) + 1
                if req is not None and key == "posted":
                    ucp.ucp_request_cancel(worker, ctypes.c_void_p(req))
                    for _ in range(10):
                        ucp.ucp_worker_progress(worker)

        if mode == "ucx_thread":
            cu.cuCtxSetCurrent(None)
            t = threading.Thread(target=post_all)
            t.start()
            t.join()
        else:
            post_all()
        print("ucx results:", results, flush=True)
        return


def run_child(libcuda_variant, mode, case, iters):
    env = dict(os.environ)
    if libcuda_variant == "host":
        hosts = [
            p
            for p in glob.glob("/usr/lib/*/libcuda.so.*")
            + glob.glob("/usr/lib64/libcuda.so.*")
            if "compat" not in p
        ]
        if not hosts:
            return None
        env["LD_PRELOAD"] = os.path.realpath(sorted(hosts)[0])
        parts = [
            x
            for x in env.get("LD_LIBRARY_PATH", "").split(":")
            if x and "compat" not in x
        ]
        env["LD_LIBRARY_PATH"] = ":".join(parts)
    cmd = [
        sys.executable,
        "-I",
        os.path.abspath(__file__),
        "child",
        mode,
        case,
        str(iters),
    ]
    try:
        r = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired as e:
        return "TIMEOUT", (e.stdout or b"").decode() if isinstance(
            e.stdout, bytes
        ) else (e.stdout or "")
    out = r.stdout.strip().splitlines()
    return r.returncode, "\n    ".join(
        out[-4:] + r.stderr.strip().splitlines()[-4:]
    )


def main():
    iters = int(os.environ.get("REPRO_ITERS", "200"))
    variants = os.environ.get("REPRO_LIBCUDA", "default,host").split(",")
    modes = os.environ.get("REPRO_MODES", ",".join(MODES)).split(",")
    for variant in variants:
        for mode in modes:
            for case in CASES:
                res = run_child(variant, mode, case, iters)
                if res is None:
                    print(
                        f"[{variant}] {mode:15s} {case:22s} SKIPPED (no host libcuda found)",
                        flush=True,
                    )
                    continue
                rc, out = res
                tag = (
                    f"signal {-rc}"
                    if isinstance(rc, int) and rc < 0
                    else f"rc={rc}"
                )
                print(
                    f"[{variant}] {mode:15s} {case:22s} {tag}\n    {out}",
                    flush=True,
                )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
