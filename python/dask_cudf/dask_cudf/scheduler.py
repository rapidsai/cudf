"""
This is a copy of the scheduler in Dask with modifications that targets cuDF
<https://github.com/dask/dask/blob/2.15.0/dask/local.py>

Asynchronous Shared-Memory Scheduler for Dask Graphs.

This scheduler coordinates several workers to execute tasks in a dask graph in
parallel.  It depends on an apply_async function as would be found in thread or
process Pools and a corresponding Queue for worker-to-scheduler communication.

It tries to execute tasks in an order which maintains a small memory footprint
throughout execution.  It does this by running tasks that allow us to release
data resources.


Task Selection Policy
=====================

When we complete a task we add more data in to our set of available data; this
new data makes new tasks available.  We preferentially choose tasks that were
just made available in a last-in-first-out fashion.  We implement this as a
simple stack.  This results in more depth-first rather than breadth first
behavior which encourages us to process batches of data to completion before
starting in on new data when possible.

When the addition of new data readies multiple tasks simultaneously we add
tasks to the stack in sorted order so that tasks with greater keynames are run
first.  This can be handy to break ties in a predictable fashion.


State
=====

Many functions pass around a ``state`` variable that holds the current state of
the computation.  This variable consists of several other dictionaries and
sets, explained below.

Constant state
--------------

1.  dependencies: {x: [a, b ,c]} a,b,c, must be run before x
2.  dependents: {a: [x, y]} a must run before x or y

Changing state
--------------

### Data

1.  cache: available concrete data.  {key: actual-data}
2.  released: data that we've seen, used, and released because it is no longer
    needed

### Jobs

1.  ready: A fifo stack of ready-to-run tasks
2.  running: A set of tasks currently in execution
3.  finished: A set of finished tasks
4.  waiting: which tasks are still waiting on others :: {key: {keys}}
    Real-time equivalent of dependencies
5.  waiting_data: available data to yet-to-be-run-tasks :: {key: {keys}}
    Real-time equivalent of dependents


Examples
--------

>>> import pprint  # doctest: +SKIP
>>> dsk = {'x': 1, 'y': 2, 'z': (inc, 'x'), 'w': (add, 'z', 'y')}
>>> pprint.pprint(start_state_from_dask(dsk))  # doctest: +SKIP
{'cache': {'x': 1, 'y': 2},
 'dependencies': {'w': {'z', 'y'}, 'x': set(), 'y': set(), 'z': {'x'}},
 'dependents': {'w': set(), 'x': {'z'}, 'y': {'w'}, 'z': {'w'}},
 'finished': set(),
 'ready': ['z'],
 'released': set(),
 'running': set(),
 'waiting': {'w': {'z'}},
 'waiting_data': {'x': {'z'}, 'y': {'w'}, 'z': {'w'}}}

Optimizations
=============

We build this scheduler with out-of-core array operations in mind.  To this end
we have encoded some particular optimizations.

Compute to release data
-----------------------

When we choose a new task to execute we often have many options.  Policies at
this stage are cheap and can significantly impact performance.  One could
imagine policies that expose parallelism, drive towards a particular output,
etc..

Our current policy is to run tasks that were most recently made available.


Inlining computations
---------------------

We hold on to intermediate computations either in memory or on disk.

For very cheap computations that may emit new copies of the data, like
``np.transpose`` or possibly even ``x + 1`` we choose not to store these as
separate pieces of data / tasks.  Instead we combine them with the computations
that require them.  This may result in repeated computation but saves
significantly on space and computation complexity.

See the function ``inline_functions`` for more information.
"""
from queue import Queue

from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies
from dask.local import (
    apply_sync,
    default_get_id,
    default_pack_exception,
    execute_task,
    finish_task,
    identity,
    nested_get,
    queue_get,
    reraise,
    start_state_from_dask,
)
from dask.utils_test import add, inc  # noqa: F401

from .order import order


"""
Task Selection
--------------

We often have a choice among many tasks to run next.  This choice is both
cheap and can significantly impact performance.

We currently select tasks that have recently been made ready.  We hope that
this first-in-first-out policy reduces memory footprint
"""

"""
`get`
-----

The main function of the scheduler.  Get is the main entry point.
"""


def get_async(
    apply_async,
    num_workers,
    dsk,
    result,
    cache=None,
    get_id=default_get_id,
    rerun_exceptions_locally=None,
    pack_exception=default_pack_exception,
    raise_exception=reraise,
    callbacks=None,
    dumps=identity,
    loads=identity,
    **kwargs,
):
    """ Asynchronous get function

    This is a general version of various asynchronous schedulers for dask.  It
    takes a an apply_async function as found on Pool objects to form a more
    specific ``get`` method that walks through the dask array with parallel
    workers, avoiding repeat computation and minimizing memory use.

    Parameters
    ----------
    apply_async : function
        Asynchronous apply function as found on Pool or ThreadPool
    num_workers : int
        The number of active tasks we should have at any one time
    dsk : dict
        A dask dictionary specifying a workflow
    result : key or list of keys
        Keys corresponding to desired data
    cache : dict-like, optional
        Temporary storage of results
    get_id : callable, optional
        Function to return the worker id, takes no arguments. Examples are
        `threading.current_thread` and `multiprocessing.current_process`.
    rerun_exceptions_locally : bool, optional
        Whether to rerun failing tasks in local process to enable debugging
        (False by default)
    pack_exception : callable, optional
        Function to take an exception and ``dumps`` method, and return a
        serialized tuple of ``(exception, traceback)`` to send back to the
        scheduler. Default is to just raise the exception.
    raise_exception : callable, optional
        Function that takes an exception and a traceback, and raises an error.
    dumps: callable, optional
        Function to serialize task data and results to communicate between
        worker and parent.  Defaults to identity.
    loads: callable, optional
        Inverse function of `dumps`.  Defaults to identity.
    callbacks : tuple or list of tuples, optional
        Callbacks are passed in as tuples of length 5. Multiple sets of
        callbacks may be passed in as a list of tuples. For more information,
        see the dask.diagnostics documentation.

    See Also
    --------
    threaded.get
    """
    queue = Queue()

    if isinstance(result, list):
        result_flat = set(flatten(result))
    else:
        result_flat = set([result])
    results = set(result_flat)

    dsk = dict(dsk)
    with local_callbacks(callbacks) as callbacks:
        _, _, pretask_cbs, posttask_cbs, _ = unpack_callbacks(callbacks)
        started_cbs = []
        succeeded = False
        # if start_state_from_dask fails, we will have something
        # to pass to the final block.
        state = {}
        try:
            for cb in callbacks:
                if cb[0]:
                    cb[0](dsk)
                started_cbs.append(cb)

            keyorder = order(dsk)
            state = start_state_from_dask(
                dsk, cache=cache, sortkey=keyorder.get
            )

            for _, start_state, _, _, _ in callbacks:
                if start_state:
                    start_state(dsk, state)

            if rerun_exceptions_locally is None:
                rerun_exceptions_locally = config.get(
                    "rerun_exceptions_locally", False
                )

            if state["waiting"] and not state["ready"]:
                raise ValueError("Found no accessible jobs in dask")

            def fire_task():
                """ Fire off a task to the thread pool """
                # Choose a good task to compute
                key = state["ready"].pop()
                state["running"].add(key)
                for f in pretask_cbs:
                    f(key, dsk, state)

                # Prep data to send
                data = dict(
                    (dep, state["cache"][dep])
                    for dep in get_dependencies(dsk, key)
                )
                # Submit
                apply_async(
                    execute_task,
                    args=(
                        key,
                        dumps((dsk[key], data)),
                        dumps,
                        loads,
                        get_id,
                        pack_exception,
                    ),
                    callback=queue.put,
                )

            # Seed initial tasks into the thread pool
            while state["ready"] and len(state["running"]) < num_workers:
                fire_task()

            # Main loop, wait on tasks to finish, insert new ones
            while state["waiting"] or state["ready"] or state["running"]:
                key, res_info, failed = queue_get(queue)
                if failed:
                    exc, tb = loads(res_info)
                    if rerun_exceptions_locally:
                        data = dict(
                            (dep, state["cache"][dep])
                            for dep in get_dependencies(dsk, key)
                        )
                        task = dsk[key]
                        _execute_task(task, data)  # Re-execute locally
                    else:
                        raise_exception(exc, tb)
                res, worker_id = loads(res_info)
                state["cache"][key] = res
                finish_task(dsk, key, state, results, keyorder.get)
                for f in posttask_cbs:
                    f(key, res, dsk, state, worker_id)

                while state["ready"] and len(state["running"]) < num_workers:
                    fire_task()

            succeeded = True

        finally:
            for _, _, _, _, finish in started_cbs:
                if finish:
                    finish(dsk, state, not succeeded)

    return nested_get(result, state["cache"])


def get_sync(dsk, keys, **kwargs):
    """A naive synchronous version of get_async

    Can be useful for debugging.
    """
    kwargs.pop("num_workers", None)  # if num_workers present, remove it
    return get_async(apply_sync, 1, dsk, keys, **kwargs)
