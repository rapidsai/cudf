/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "tests_common.h"


size_t present_count(const orc_bitmap* present, size_t present_size)
{
    int expected_count = 0;

    for (int i = 0; i < present_size; i++) {
        expected_count += getBitCount(present[i]);
    }

    return expected_count;
}

const orc_bitmap* gen_present(size_t& present_length, size_t valid_count)
{
    orc_bitmap* present;

    int the_unit = std::max<int>(valid_count >> 3, 4);
    int mult = 3;
    int the_count;

    present_length = the_unit * mult;
    present = (orc_bitmap*)malloc(present_length);
    set_random<orc_bitmap>(present, present_length, 0x00, 0xff);
    the_count = present_count(present, present_length);

    while (the_count < valid_count)
    {
        mult++;

        present_length = the_unit * mult;
        present = (orc_bitmap*)realloc(present, present_length);
        orc_bitmap* the_new = present + the_unit * (mult - 1);
        set_random<orc_bitmap>(the_new, the_unit, 0x00, 0xff);
        the_count += present_count(the_new, the_unit);

        assert(mult < 20);

    }

    return present;
}
