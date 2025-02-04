# Copyright (c) 2025, NVIDIA CORPORATION.

import queue
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait


# Function to be called in each thread
def thread_function(index):
    # Import the library in the thread
    import pandas as pd

    x = pd.Series([1, 2, 3])

    return f"{index}" + str(type(type(x)))


def main():
    # Number of threads to use
    num_threads = 4

    # Queue of tasks to be processed by the threads
    task_queue = queue.Queue()
    for i in range(num_threads):
        task_queue.put((i,))

    # List to hold the futures
    futures = []

    # Use ThreadPoolExecutor to manage the threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        while not task_queue.empty():
            task = task_queue.get()
            future = executor.submit(thread_function, *task)
            futures.append(future)

    # Wait for all threads to complete
    _, _ = wait(futures, return_when=ALL_COMPLETED)

    # Process the results
    for i, future in enumerate(futures):
        result = future.result()
        print(f"Result from thread {i + 1}: {result}")


if __name__ == "__main__":
    from cudf.pandas.module_accelerator import disable_module_accelerator

    with disable_module_accelerator():
        main()
