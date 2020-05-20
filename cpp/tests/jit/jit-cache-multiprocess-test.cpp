/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <sys/types.h>
#include <unistd.h>
#include "jit-cache-test.hpp"

#if defined(JITIFY_USE_CACHE)
/**
 * @brief This test runs two processes that try to access the same kernel
 *
 * This is a stress test.
 *
 * A single test process is forked before invocation of CUDA and then both the
 * parent and child processes try to get and run a kernel. The child process
 * clears the cache before each iteration of the test so that the cache has to
 * be re-written by it. The parent process runs on a changing time offset so
 * that it sometimes gets the kernel from cache and sometimes it doesn't.
 *
 * The aim of this test is to check that the file cache doesn't get corrupted
 * when multiple processes are reading/writing to it at the same time. Since
 * the public API of JitCache doesn't return the serialized string of the
 * cached kernel, the way to test its validity is to run it on test data.
 **/
TEST_F(JitCacheMultiProcessTest, MultiProcessTest)
{
  int num_tests = 20;
  int *input, *output;
  int expect = 64;

  auto tester = [&](int pid, int test_no) {
    // Brand new cache object that has nothing in in-memory cache
    cudf::jit::cudfJitCache cache;

    // Parent writes to output[0], child writes to output[1]
    size_t idx = 2 * test_no + (pid == 0);

    *input      = 4;
    output[idx] = 1;

    // make program
    auto program = cache.getProgram("FileCacheTestProg3", program3_source);
    // make kernel
    auto kernel = cache.getKernelInstantiation("my_kernel", program, {"3", "int"});
    (*std::get<1>(kernel)).configure(grid, block).launch(input, &output[idx]);
    CUDA_TRY(cudaDeviceSynchronize());

    ASSERT_TRUE(expect == output[idx]) << "Expected val: " << expect << '\n'
                                       << "  Actual val: " << output[idx];
  };

  // This pipe is how the child process will send output to parent
  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1) << "Unable to create pipe";

  pid_t cpid = fork();
  ASSERT_TRUE(cpid >= 0) << "Fork failed";

  if (cpid > 0) {      // Parent
    close(pipefd[1]);  // Close write end of pipe. Parent doesn't write.
    usleep(100000);
  } else {                           // Child
    close(pipefd[0]);                // Close read end of pipe. Child doesn't read.
    dup2(pipefd[1], STDOUT_FILENO);  // redirect stdout to pipe
  }

  CUDA_TRY(cudaMallocManaged(&input, sizeof(input)));
  CUDA_TRY(cudaMallocManaged(&output, sizeof(output) * num_tests * 2));

  for (size_t i = 0; i < num_tests; i++) {
    if (cpid > 0)
      usleep(10000);
    else
      purgeFileCache();

    tester(cpid, i);
  }

  // Child ends here --------------------------------------------------------

  if (cpid > 0) {
    int status;
    wait(&status);

    std::cout << "Child output begin:" << std::endl;
    char buf;
    while (read(pipefd[0], &buf, 1) > 0) ASSERT_EQ(write(STDOUT_FILENO, &buf, 1), 1);
    ASSERT_EQ(write(STDOUT_FILENO, "\n", 1), 1);
    std::cout << "Child output end" << std::endl;

    ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit normally.";
    ASSERT_EQ(WEXITSTATUS(status), 0) << "Error in child.";
  }
}
#endif

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  // This test relies on the fact that the cuda context will be created in
  // each process separately after the fork. Using rmm_mode=pool will cause
  // the cuda context to be created at rmmInitialize, which happens before
  // the fork. So we hardcode the rmm_mode to "cuda"
  auto const rmm_mode = "cuda";
  auto resource       = cudf::test::create_memory_resource(rmm_mode);
  rmm::mr::set_default_resource(resource.get());
  return RUN_ALL_TESTS();
}
