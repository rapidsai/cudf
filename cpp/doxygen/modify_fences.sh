#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION.

# This script modifies the GitHub Markdown style code fences in our MD files
# into the PHP style that Doxygen supports, allowing us to display code
# properly both on the GitHub GUI and in published Doxygen documentation.

sed 's/```c++/```{.cpp}/g' "$@"
