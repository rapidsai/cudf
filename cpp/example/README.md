# Basic Standalone libcudf C++ application

This simple C++ example demonstrates a basic libcudf use case and provides a
minimal example of building your own application based on libcudf using CMake.

The example source code loads a csv file that contains stock prices from 4
companies spanning across 5 days, computes the average of the closing price
for each company and writes the result in csv format.

## How to compile and execute

The compilation process is automated by a Dockerfile included in the project.

Prerequisites:
- docker (API >= 1.40 to support --gpus)
- nvidia driver >= 450.80.02 (to support cudatoolkit 11.1)

### Step 1: build environment in docker
```bash
docker build . -t rapidsenv
```

### Step 2: start the container
```bash
docker run -t -d -v $PWD:/workspace --gpus all --name rapidsenvrt rapidsenv
```

### (When active container running) Configure project
```bash
docker exec rapidsenvrt sh -c "cmake -S . -B build/"
```

### (When active container running) Build project
```bash
docker exec rapidsenvrt sh -c "cmake --build build/ --parallel $PARALLEL_LEVEL"
```
The first time running this command will take a long time because it will build
libcudf on the host machine. It may be sped up by configuring the proper
`PARALLEL_LEVEL` number.

### (When active container running) Execute binary
```bash
docker exec rapidsenvrt sh -c "build/libcudf_example"
```
