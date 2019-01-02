#!/usr/bin/env bash
# Reports relevant environment information useful for diagnosing and
# debugging cuDF issues.
# Usage: 
# "./print_env.sh" - prints to stdout
# "./print_env.sh > env.txt" - prints to file "env.txt"

echo "**git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
git log --decorate -n 1
else
echo "Not inside a git repository"
fi
echo 

echo "***OS Information***"
cat /etc/*-release
uname -a
echo 

echo "***GPU Information***"
nvidia-smi
echo 

echo "***CPU***"
lscpu
echo

echo "***CMake***"
which cmake && cmake --version 
echo 

echo "***g++***"
which g++ && g++ --version 
echo 

echo "***nvcc***"
which nvcc && nvcc --version 
echo 

echo "***Python***"
which python && python --version
echo

echo "***Environment Variables***"

printf '%-32s: %s\n' PATH $PATH

printf '%-32s: %s\n' LD_LIBRARY_PATH $LD_LIBRARY_PATH

printf '%-32s: %s\n' NUMBAPRO_NVVM $NUMBAPRO_NVVM

printf '%-32s: %s\n' NUMBAPRO_LIBDEVICE $NUMBAPRO_LIBDEVICE

printf '%-32s: %s\n' CONDA_PREFIX $CONDA_PREFIX

printf '%-32s: %s\n' PYTHON_PATH $PYTHON_PATH

echo


# Print conda packages if conda exists
if type "conda" &> /dev/null; then
echo '***conda packages***'
which conda && conda list 
echo
# Print pip packages if pip exists
elif type "pip" &> /dev/null; then
echo "conda not found"
echo "***pip packages***"
which pip && pip list
echo
else
echo "conda not found"
echo "pip not found"
fi
