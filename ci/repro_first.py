# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ctypes
import os
import subprocess
import sys
import threading
import time

PTX = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry k() { ret; }
"""
MODES = ["driver", "ucx"]
PROCS = int(os.environ.get("REPRO_PROCS", "5"))
ROUNDS = int(os.environ.get("REPRO_ROUNDS", "25"))


def ck(res, what):
    if res != 0:
        print(f"{what} failed: {res}", flush=True)
        os._exit(3)


def gpu_work(cu, ctx):
    cu.cuCtxSetCurrent(ctx)
    stream = ctypes.c_void_p()
    ck(cu.cuStreamCreate(ctypes.byref(stream), 0), "stream")
    mod = ctypes.c_void_p()
    func = ctypes.c_void_p()
    if cu.cuModuleLoadData(ctypes.byref(mod), PTX) == 0:
        cu.cuModuleGetFunction(ctypes.byref(func), mod, b"k")
    for i in range(20):
        p = ctypes.c_uint64()
        cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(1 << (16 + i % 6)))
        cu.cuMemsetD8Async(p, 0, ctypes.c_size_t(1 << 16), stream)
        if func.value:
            cu.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, None, None)
        cu.cuStreamSynchronize(stream)
        cu.cuMemFree_v2(p)


def child(mode, start_at):
    cu = ctypes.CDLL("libcuda.so.1")
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "device")
    ctx = ctypes.c_void_p()
    ck(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "retain")
    ck(cu.cuCtxSetCurrent(ctx), "setcurrent")
    buf = ctypes.c_uint64()
    ck(cu.cuMemAlloc_v2(ctypes.byref(buf), ctypes.c_size_t(1 << 20)), "alloc")
    page = os.sysconf("SC_PAGE_SIZE")

    if mode == "ucx":
        ucp = ctypes.CDLL("libucp.so.0")

        class Params(ctypes.Structure):
            _fields_ = [
                ("field_mask", ctypes.c_uint64),
                ("features", ctypes.c_uint64),
                ("request_size", ctypes.c_size_t),
                ("pad", ctypes.c_char * 512),
            ]

        class Pad(ctypes.Structure):
            _fields_ = [("pad", ctypes.c_char * 512)]

        maj, mnr = ctypes.c_uint(), ctypes.c_uint()
        ucp.ucp_get_version(
            ctypes.byref(maj), ctypes.byref(mnr), ctypes.byref(ctypes.c_uint())
        )
        cfg = ctypes.c_void_p()
        ck(ucp.ucp_config_read(None, None, ctypes.byref(cfg)), "config")
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
        worker = ctypes.c_void_p()
        ck(
            ucp.ucp_worker_create(
                context, ctypes.byref(Pad()), ctypes.byref(worker)
            ),
            "worker",
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

        def first_call():
            ucp.ucp_tag_recv_nbx(
                worker,
                buf.value,
                1 << 20,
                1,
                0xFFFFFFFFFFFFFFFF,
                ctypes.byref(Pad()),
            )
    else:
        fn = ctypes.c_void_p()
        status = ctypes.c_int()
        ck(
            cu.cuGetProcAddress_v2(
                b"cuMemGetHandleForAddressRange",
                ctypes.byref(fn),
                ctypes.c_int(12000),
                ctypes.c_ulonglong(0),
                ctypes.byref(status),
            ),
            "getprocaddress",
        )
        proto = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_ulonglong,
        )
        get_handle = proto(fn.value)

        def first_call():
            fd = ctypes.c_int(-1)
            start = buf.value & ~(page - 1)
            if get_handle(ctypes.byref(fd), start, 1 << 20, 1, 0) == 0:
                os.close(fd.value)

    worker_thread = threading.Thread(target=gpu_work, args=(cu, ctx))
    while time.time() < start_at:
        pass
    worker_thread.start()
    first_call()
    worker_thread.join()
    print("ok", flush=True)


def main():
    totals = {}
    for rnd in range(ROUNDS):
        for mode in MODES:
            start_at = time.time() + 4.0
            procs = [
                subprocess.Popen(
                    [
                        sys.executable,
                        "-I",
                        os.path.abspath(__file__),
                        "child",
                        mode,
                        repr(start_at),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for _ in range(PROCS)
            ]
            codes = []
            for pr in procs:
                try:
                    pr.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    pr.kill()
                    pr.wait()
                codes.append(pr.returncode)
            crashed = sum(1 for c in codes if c is not None and c != 0)
            key = f"{mode}"
            t = totals.setdefault(key, [0, 0, 0])
            t[0] += len(codes)
            t[1] += crashed
            t[2] += 1 if crashed else 0
            if crashed:
                print(f"round {rnd} {mode}: return codes {codes}", flush=True)
    for k, (n, c, r) in totals.items():
        print(
            f"{k}: {c}/{n} processes failed, {r}/{ROUNDS} rounds with a failure",
            flush=True,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(sys.argv[2], float(sys.argv[3]))
    else:
        main()
