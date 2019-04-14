## Purpose

This script is designed for developer and contributor use. This tool mimics the actions of gpuCI on your local machine. This allows you to test and even debug your code inside a gpuCI base container before pushing your code as a GitHub commit.
The script can be helpful in locally triaging and debugging RAPIDS continuous integration failures.

## Usage

```
bash build.sh [-h] [-H] [-s] [-r <repo_dir>] [-i <image_name>]
Build and test your local repository using a base gpuCI Docker image

where:
    -H   Show this help text
    -r   Path to repository (defaults to working directory)
    -i   Use Docker image (default is gpuci/rapidsai-base:cuda10.0-ubuntu16.04-gcc5-py3.6)
    -s   Skip building and testing and start an interactive shell in a container of the Docker image
```

Example Usage:
`bash build.sh -r ~/rapids/cudf -i gpuci/cuda9.2-ubuntu16.04-gcc5-py3.6`

For a full list of available gpuCI docker images, visit our [DockerHub](https://hub.docker.com/r/gpuci/rapidsai-base/tags) page.


## Information

There are some caveats to be aware of when using this script, especially if you plan on developing from within the container itself.


### Docker Image Build Repository

The docker image will generate build artifacts in a folder on your machine located in the `root` directory of the repository you passed to the script. For the above example, the directory is named `~/rapids/cudf/build_rapidsai-base_cuda9.2-ubuntu16.04-gcc5-py3.6/`. Feel free to remove this directory after the script is finished.

*Note*: The script *will not* override your local build repository. Your local environment stays in tact.


### Where The User is Dumped

The script will build your repository and run all tests. If any tests fail, it dumps the user into the docker container itself to allow you to debug from within the container. If all the tests pass as expected the container exits and is automatically removed. Remember to exit the container if tests fail and you do not wish to debug within the container itself.


### Container File Structure

Your repository will be located in the `/rapids/` folder of the container. This folder is volume mounted from the local machine. Any changes to the code in this repository are replicated onto the local machine. The `cpp/build` directory within your repository is on a separate mount to avoid conflicting with your local build artifacts. The file within the `cpp/build` directory in the container named `build.sh` is a small script that builds your repository from source and run tests.
