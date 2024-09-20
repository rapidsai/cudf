#!/usr/bin/env bash

set -x

# Install cuFile
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/ /" | sudo tee /etc/apt/sources.list.d/cuda-ubuntu2204-sbsa.list
sudo apt update
sudo apt install -y libcufile-dev-12-5 libcufile-12-5
rm cuda-keyring_1.1-1_all.deb
