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

#include "jit-cache-test.hpp"
#include <sys/types.h>
#include <unistd.h>


#if defined(JITIFY_USE_CACHE)
TEST_F(JitCacheMultiProcessTest, MultiProcessTest) {

    auto tester = [&] () {
        // Brand new cache object that has nothing in in-memory cache
        cudf::jit::cudfJitCache cache;
        
        // Single value column
        auto column = cudf::test::column_wrapper<int>{{4,0}};
        auto expect = cudf::test::column_wrapper<int>{{64,0}};

        // make program
        auto program = cache.getProgram("MemoryCacheTestProg", program_source);
        // make kernel
        auto kernel = cache.getKernelInstantiation("my_kernel",
                                                    program,
                                                    {"3", "int"});
        (*std::get<1>(kernel)).configure_1d_max_occupancy()
                 .launch(column.get()->data);

        ASSERT_TRUE(expect == column) << "Expected col: " << expect.to_str()
                                      << "  Actual col: " << column.to_str();

    };

    // This pipe is how the child process will send output to parent
    int pipefd[2];
    ASSERT_NE(pipe(pipefd), -1) << "Unable to create pipe";

    pid_t cpid = fork();
    ASSERT_TRUE( cpid >= 0 ) << "Fork failed";
    
    if (cpid > 0) { // Parent
        close(pipefd[1]); // Close write end of pipe. Parent doesn't write.
        usleep(100000);
    }
    else { // Child
        close(pipefd[0]); // Close read end of pipe. Child doesn't read.
        dup2(pipefd[1], STDOUT_FILENO); // redirect stdout to pipe
    }

    for (size_t i = 0; i < 20; i++)
    {
        if (cpid > 0) usleep(10000);
        else purgeFileCache();

        tester();
    }

    // Child ends here --------------------------------------------------------

    if (cpid > 0) {
        int status;
        wait(&status);

        std::cout << "Child output begin:" << std::endl;
        char buf;
        while (read(pipefd[0], &buf, 1) > 0)
            ASSERT_EQ(write(STDOUT_FILENO, &buf, 1), 1);
        ASSERT_EQ(write(STDOUT_FILENO, "\n", 1), 1);
        std::cout << "Child output end" << std::endl;

        ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit normally.";
        ASSERT_EQ(WEXITSTATUS(status), 0) << "Error in child.";
    }
}
#endif
