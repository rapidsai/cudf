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
which cmake && cmake --version | grep "version"
echo " "

echo "***which g++ && g++ --version***"
which g++ && g++ --version | grep "g++"
echo " "

echo "***which nvcc && nvcc --version***"
which nvcc && nvcc --version | grep "release"
echo " "

echo "***echo NUMBAPRO_NVVM***"
echo $NUMBAPRO_NVVM
echo " "

echo "***echo NUMBAPRO_LIBDEVICE***"
echo $NUMBAPRO_LIBDEVICE
echo " "




