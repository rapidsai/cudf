# Copyright (c) 2018, NVIDIA CORPORATION.

import collections
import logging
import multiprocessing
import pickle
import socket
import threading

from numba import cuda

import rmm

try:
    import zmq

    _HAVE_ZMQ = True
except ImportError:
    _HAVE_ZMQ = False


logger = logging.getLogger(__name__)


_global_port = [None]
_global_addr = [None]
_server_started = False
_server_lock = threading.Lock()

_USING_IPC = False


def is_using_ipc():
    return _USING_IPC


def enable_ipc():
    global _USING_IPC
    if not _HAVE_ZMQ:
        raise ImportError("can't import zmq")
    _USING_IPC = True
    start_server()


class ObjCache(object):
    def __init__(self):
        self._dct = {}
        self._ipch = {}
        self._refct = collections.defaultdict(int)

    def set(self, key, value):
        with _server_lock:
            self._refct[key] += 1
            if self._refct[key] == 1:
                self._dct[key] = value

    def get_ipc(self, key):
        with _server_lock:
            if key in self._ipch:
                ipch = self._ipch[key]
            else:
                ipch = self._dct[key].get_ipc_handle()
                self._ipch[key] = ipch
            return ipch

    def get(self, key):
        with _server_lock:
            return self._dct[key]

    def drop(self, key):
        with _server_lock:
            self._refct[key] -= 1
            refct = self._refct[key]
            if refct == 0:
                if key in self._ipch:
                    del self._ipch[key]

                del self._refct[key]
                del self._dct[key]


_out_cache = ObjCache()


def init_server():
    _global_addr[0] = socket.gethostname()
    logger.info("host addr: %s", _global_addr[0])
    devnum = cuda.get_current_device().id
    th = threading.Thread(target=server_loop, args=[devnum])
    th.daemon = True
    th.start()


def server_loop(devnum):
    logger.info("server loop starts")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    selected_port = socket.bind_to_random_port("tcp://*")
    _global_port[0] = selected_port
    logger.info("bind to port: %s", selected_port)

    with cuda.gpus[devnum]:
        while True:
            req = socket.recv()
            out = _handle_request(req)
            socket.send(out)


def serialize_gpu_data(gpu_data):
    context = _get_context()
    remoteinfo = _global_addr[0], _global_port[0]
    key = _get_key(gpu_data)

    logger.debug("cache key: %s", key)
    _out_cache.set(key, gpu_data)

    header = {"key": key, "context": context, "remoteinfo": remoteinfo}
    frames = []
    return header, frames


def rebuild_gpu_data(context, key, remoteinfo):
    if context != _get_context():
        logger.debug("do transfer: %s", key)
        out = _request_transfer(key, remoteinfo)
        return out

    else:
        logger.debug("same process: %s", key)
        return _out_cache.get(key)


def _get_context():
    pid = multiprocessing.current_process().pid
    ctxid = cuda.current_context().handle.value
    return pid, ctxid


def _hash_ipc_handle(ipchandle):
    return str(hex(hash(tuple(ipchandle._ipc_handle.handle)))).encode()


def _get_key(gpudata):
    return str(hash(gpudata)).encode()


def _handle_request(req):
    method, key = pickle.loads(req)
    data = _out_cache.get(key)
    if method == "NET":
        # NET
        return pickle.dumps(data.copy_to_host())
    elif method == "IPC":
        # IPC
        out = pickle.dumps(_out_cache.get_ipc(key))
        return out
    elif method == "DROP":
        # DROP
        _out_cache.drop(key)
        return pickle.dumps("OK")
    else:
        raise NotImplementedError("unknown method {!r}".format(method))


def _request_transfer(key, remoteinfo):
    logger.info("rebuild from: %s for %r", remoteinfo, key)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{0}:{1}".format(*remoteinfo))

    myaddr = _global_addr[0]
    theiraddr = remoteinfo[0]
    if myaddr == theiraddr:
        # Same machine go by IPC
        logger.info("request by IPC")
        socket.send(pickle.dumps(("IPC", key)))
        rcv = socket.recv()
        ipch = pickle.loads(rcv)
        # Open IPC and copy to local context

        with ipch as data:
            copied = rmm.device_array_like(data)
            copied.copy_to_device(data)

        # Release
        _request_drop(socket, key)
        return copied
    else:
        # Different machine go by NET
        logger.info("request by NET: %s->%s", theiraddr, myaddr)
        socket.send(pickle.dumps(("NET", key)))
        rcv = socket.recv()
        output = rmm.to_device(pickle.loads(rcv))
        # Release
        _request_drop(socket, key)
        return output


def _request_drop(socket, key):
    socket.send(pickle.dumps(("DROP", key)))
    socket.recv()


def start_server():
    global _server_started
    with _server_lock:
        if _server_started:
            return
        _server_started = True
        init_server()
