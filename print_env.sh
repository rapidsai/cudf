echo "***git log --decorate -n 1***"
git log --decorate -n 1
echo " "

echo "***cat /etc/*-release***"
cat /etc/*-release
echo " "

echo "***uname -a***"
uname -a
echo " "

echo "***nvidia-smi***"
nvidia-smi
echo " "

echo '***conda list "pandas|arrow|numpy|^python$"***'
conda list 'pandas|arrow|numpy|^python$'
echo " "

echo "***which cmake && cmake --version***"
which cmake && cmake --version
echo " "

echo "***which g++ && g++ --version***"
which g++ && g++ --version
echo " "

echo "***which nvcc && nvcc --version***"
which nvcc && nvcc --version




