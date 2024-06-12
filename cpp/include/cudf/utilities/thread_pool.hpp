/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

/**
 * Modified from https://github.com/bshoshany/thread-pool
 * @copyright Copyright (c) 2021 Barak Shoshany. Licensed under the MIT license.
 *            See file LICENSE for detail or copy at https://opensource.org/licenses/MIT
 */

#include <atomic>       // std::atomic
#include <chrono>       // std::chrono
#include <cstdint>      // std::int_fast64_t, std::uint_fast32_t
#include <functional>   // std::function
#include <future>       // std::future, std::promise
#include <memory>       // std::shared_ptr, std::unique_ptr
#include <mutex>        // std::mutex, std::scoped_lock
#include <queue>        // std::queue
#include <thread>       // std::this_thread, std::thread
#include <type_traits>  // std::decay_t, std::enable_if_t, std::is_void_v, std::invoke_result_t
#include <utility>      // std::move, std::swap

namespace cudf {
namespace detail {

/**
 * @brief A C++17 thread pool class. The user submits tasks to be executed into a queue. Whenever a
 * thread becomes available, it pops a task from the queue and executes it. Each task is
 * automatically assigned a future, which can be used to wait for the task to finish executing
 * and/or obtain its eventual return value.
 */
class thread_pool {
  using ui32 = int;

 public:
  /**
   * @brief Construct a new thread pool.
   *
   * @param _thread_count The number of threads to use. The default value is the total number of
   * hardware threads available, as reported by the implementation. With a hyperthreaded CPU, this
   * will be twice the number of CPU cores. If the argument is zero, the default value will be used
   * instead.
   */
  thread_pool(ui32 const& _thread_count = std::thread::hardware_concurrency())
    : thread_count(_thread_count ? _thread_count : std::thread::hardware_concurrency()),
      threads(new std::thread[_thread_count ? _thread_count : std::thread::hardware_concurrency()])
  {
    create_threads();
  }

  /**
   * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads.
   * Note that if the variable paused is set to true, then any tasks still in the queue will never
   * be executed.
   */
  ~thread_pool()
  {
    wait_for_tasks();
    running = false;
    destroy_threads();
  }

  /**
   * @brief Get the number of tasks currently waiting in the queue to be executed by the threads.
   *
   * @return The number of queued tasks.
   */
  [[nodiscard]] size_t get_tasks_queued() const
  {
    std::scoped_lock const lock(queue_mutex);
    return tasks.size();
  }

  /**
   * @brief Get the number of tasks currently being executed by the threads.
   *
   * @return The number of running tasks.
   */
  [[nodiscard]] ui32 get_tasks_running() const { return tasks_total - (ui32)get_tasks_queued(); }

  /**
   * @brief Get the total number of unfinished tasks - either still in the queue, or running in a
   * thread.
   *
   * @return The total number of tasks.
   */
  [[nodiscard]] ui32 get_tasks_total() const { return tasks_total; }

  /**
   * @brief Get the number of threads in the pool.
   *
   * @return The number of threads.
   */
  [[nodiscard]] ui32 get_thread_count() const { return thread_count; }

  /**
   * @brief Parallelize a loop by splitting it into blocks, submitting each block separately to the
   * thread pool, and waiting for all blocks to finish executing. The loop will be equivalent to:
   * for (T i = first_index; i <= last_index; i++) loop(i);
   *
   * @tparam T The type of the loop index. Should be a signed or unsigned integer.
   * @tparam F The type of the function to loop through.
   * @param first_index The first index in the loop (inclusive).
   * @param last_index The last index in the loop (inclusive).
   * @param loop The function to loop through. Should take exactly one argument, the loop index.
   * @param num_tasks The maximum number of tasks to split the loop into. The default is to use the
   * number of threads in the pool.
   */
  template <typename T, typename F>
  void parallelize_loop(T first_index, T last_index, F const& loop, ui32 num_tasks = 0)
  {
    if (num_tasks == 0) num_tasks = thread_count;
    if (last_index < first_index) std::swap(last_index, first_index);
    size_t total_size = last_index - first_index + 1;
    size_t block_size = total_size / num_tasks;
    if (block_size == 0) {
      block_size = 1;
      num_tasks  = (ui32)total_size > 1 ? (ui32)total_size : 1;
    }
    std::atomic<ui32> blocks_running = 0;
    for (ui32 t = 0; t < num_tasks; t++) {
      T start = (T)(t * block_size + first_index);
      T end   = (t == num_tasks - 1) ? last_index : (T)((t + 1) * block_size + first_index - 1);
      blocks_running++;
      push_task([start, end, &loop, &blocks_running] {
        for (T i = start; i <= end; i++)
          loop(i);
        blocks_running--;
      });
    }
    while (blocks_running != 0) {
      sleep_or_yield();
    }
  }

  /**
   * @brief Push a function with no arguments or return value into the task queue.
   *
   * @tparam F The type of the function.
   * @param task The function to push.
   */
  template <typename F>
  void push_task(F const& task)
  {
    tasks_total++;
    {
      std::scoped_lock const lock(queue_mutex);
      tasks.push(std::function<void()>(task));
    }
  }

  /**
   * @brief Push a function with arguments, but no return value, into the task queue.
   * @details The function is wrapped inside a lambda in order to hide the arguments, as the tasks
   * in the queue must be of type std::function<void()>, so they cannot have any arguments or return
   * value. If no arguments are provided, the other overload will be used, in order to avoid the
   * (slight) overhead of using a lambda.
   *
   * @tparam F The type of the function.
   * @tparam A The types of the arguments.
   * @param task The function to push.
   * @param args The arguments to pass to the function.
   */
  template <typename F, typename... A>
  void push_task(F const& task, A const&... args)
  {
    push_task([task, args...] { task(args...); });
  }

  /**
   * @brief Reset the number of threads in the pool. Waits for all currently running tasks to be
   * completed, then destroys all threads in the pool and creates a new thread pool with the new
   * number of threads. Any tasks that were waiting in the queue before the pool was reset will then
   * be executed by the new threads. If the pool was paused before resetting it, the new pool will
   * be paused as well.
   *
   * @param _thread_count The number of threads to use. The default value is the total number of
   * hardware threads available, as reported by the implementation. With a hyperthreaded CPU, this
   * will be twice the number of CPU cores. If the argument is zero, the default value will be used
   * instead.
   */
  void reset(ui32 const& _thread_count = std::thread::hardware_concurrency())
  {
    bool was_paused = paused;
    paused          = true;
    wait_for_tasks();
    running = false;
    destroy_threads();
    thread_count = _thread_count ? _thread_count : std::thread::hardware_concurrency();
    threads      = std::make_unique<std::thread[]>(thread_count);
    paused       = was_paused;
    create_threads();
    running = true;
  }

  /**
   * @brief Submit a function with zero or more arguments and a return value into the task queue,
   * and get a future for its eventual returned value.
   *
   * @tparam F The type of the function.
   * @tparam A The types of the zero or more arguments to pass to the function.
   * @tparam R The return type of the function.
   * @param task The function to submit.
   * @param args The zero or more arguments to pass to the function.
   * @return A future to be used later to obtain the function's returned value, waiting for it to
   * finish its execution if needed.
   */
  template <typename F,
            typename... A,
            typename R = std::invoke_result_t<std::decay_t<F>, std::decay_t<A>...>>
  std::future<R> submit(F const& task, A const&... args)
  {
    std::shared_ptr<std::promise<R>> promise(new std::promise<R>);
    std::future<R> future = promise->get_future();
    push_task([task, args..., promise] {
      try {
        if constexpr (std::is_void_v<R>) {
          task(args...);
          promise->set_value();
        } else {
          promise->set_value(task(args...));
        }
      } catch (...) {
        promise->set_exception(std::current_exception());
      };
    });
    return future;
  }

  /**
   * @brief Wait for tasks to be completed. Normally, this function waits for all tasks, both those
   * that are currently running in the threads and those that are still waiting in the queue.
   * However, if the variable paused is set to true, this function only waits for the currently
   * running tasks (otherwise it would wait forever). To wait for a specific task, use submit()
   * instead, and call the wait() member function of the generated future.
   */
  void wait_for_tasks()
  {
    while (true) {
      if (!paused) {
        if (tasks_total == 0) break;
      } else {
        if (get_tasks_running() == 0) break;
      }
      sleep_or_yield();
    }
  }

  /**
   * @brief An atomic variable indicating to the workers to pause. When set to true, the workers
   * temporarily stop popping new tasks out of the queue, although any tasks already executed will
   * keep running until they are done. Set to false again to resume popping tasks.
   */
  std::atomic<bool> paused = false;

  /**
   * @brief The duration, in microseconds, that the worker function should sleep for when it cannot
   * find any tasks in the queue. If set to 0, then instead of sleeping, the worker function will
   * execute std::this_thread::yield() if there are no tasks in the queue. The default value is
   * 1000.
   */
  ui32 sleep_duration = 1000;

 private:
  /**
   * @brief Create the threads in the pool and assign a worker to each thread.
   */
  void create_threads()
  {
    for (ui32 i = 0; i < thread_count; i++) {
      threads[i] = std::thread(&thread_pool::worker, this);
    }
  }

  /**
   * @brief Destroy the threads in the pool by joining them.
   */
  void destroy_threads()
  {
    for (ui32 i = 0; i < thread_count; i++) {
      threads[i].join();
    }
  }

  /**
   * @brief Try to pop a new task out of the queue.
   *
   * @param task A reference to the task. Will be populated with a function if the queue is not
   * empty.
   * @return true if a task was found, false if the queue is empty.
   */
  bool pop_task(std::function<void()>& task)
  {
    std::scoped_lock const lock(queue_mutex);
    if (tasks.empty())
      return false;
    else {
      task = std::move(tasks.front());
      tasks.pop();
      return true;
    }
  }

  /**
   * @brief Sleep for sleep_duration microseconds. If that variable is set to zero, yield instead.
   *
   */
  void sleep_or_yield()
  {
    if (sleep_duration)
      std::this_thread::sleep_for(std::chrono::microseconds(sleep_duration));
    else
      std::this_thread::yield();
  }

  /**
   * @brief A worker function to be assigned to each thread in the pool. Continuously pops tasks out
   * of the queue and executes them, as long as the atomic variable running is set to true.
   */
  void worker()
  {
    while (running) {
      std::function<void()> task;
      if (!paused && pop_task(task)) {
        task();
        tasks_total--;
      } else {
        sleep_or_yield();
      }
    }
  }

  /**
   * @brief A mutex to synchronize access to the task queue by different threads.
   */
  mutable std::mutex queue_mutex;

  /**
   * @brief An atomic variable indicating to the workers to keep running. When set to false, the
   * workers permanently stop working.
   */
  std::atomic<bool> running = true;

  /**
   * @brief A queue of tasks to be executed by the threads.
   */
  std::queue<std::function<void()>> tasks;

  /**
   * @brief The number of threads in the pool.
   */
  ui32 thread_count;

  /**
   * @brief A smart pointer to manage the memory allocated for the threads.
   */
  std::unique_ptr<std::thread[]> threads;

  /**
   * @brief An atomic variable to keep track of the total number of unfinished tasks - either still
   * in the queue, or running in a thread.
   */
  std::atomic<ui32> tasks_total = 0;
};

}  // namespace detail
}  // namespace cudf
