# cuDF Development Containers

This directory contains [devcontainer configurations](https://containers.dev/implementors/json_reference/) for using VSCode to [develop in a container](https://code.visualstudio.com/docs/devcontainers/containers) via the `Remote Containers` [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) or [GitHub Codespaces](https://github.com/codespaces).

This container is a turnkey development environment for building and testing the cuDF C++ and Python libraries.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Host bind mounts](#host-bind-mounts)
* [Launch a Dev Container](#launch-a-dev-container)
  * [via VSCode](#via-vscode)
  * [via `launch.sh`](#via-launchsh)
    * [Single mode](#single-mode)
    * [Unified mode](#unified-mode)
    * [Isolated mode](#isolated-mode)

## Prerequisites

* [VSCode](https://code.visualstudio.com/download)
* [VSCode Remote Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

## Host bind mounts

By default, the following directories are bind-mounted into the devcontainer:

* `${repo}:/home/coder/cudf`
* `${repo}/../.aws:/home/coder/.aws`
* `${repo}/../.local:/home/coder/.local`
* `${repo}/../.cache:/home/coder/.cache`
* `${repo}/../.conda:/home/coder/.conda`
* `${repo}/../.config:/home/coder/.config`

This ensures caches, configurations, dependencies, and your commits are persisted on the host across container runs.

## Launch a Dev Container

A devcontainer can be launched directly by VSCode, or via a custom `launch.sh` shell script.

### via VSCode

To launch a devcontainer from VSCode, open the cuDF repo and select the "Reopen in Container" button in the bottom right:<br/><img src="https://user-images.githubusercontent.com/178183/221771999-97ab29d5-e718-4e5f-b32f-2cdd51bba25c.png"/>

Alternatively, open the VSCode command palette (typically `cmd/ctrl + shift + P`) and run the "Rebuild and Reopen in Container" command.

### via `launch.sh`

Use the `.devcontainer/launch.sh` script to start a new instance of the development container and launch a fresh VSCode window connected to it.

VSCode extends its [single-window-per-folder](https://github.com/microsoft/vscode/issues/2686) process model to devcontainers. Opening the same devcontainer in separate windows doesn't create two separate containers -- instead you have two VSCode windows each connected to the same running container.

`launch.sh` takes two arguments, a `mode` and a `package manager`.

* The `mode` argument determines how the devcontainer interacts with the files on the host.
* The `package manager` argument can be either `conda`, or `pip`. This determines whether the devcontainer uses `conda` or `pip` to install the dependencies (the default is `conda`). `pip` is experimental/not working for normal dev, and is currently meant to aid in pip packaging work.

#### Single mode

`.devcontainer/launch.sh single` launches the devcontainer with the [default bind mounts](#host-bind-mounts). RMM and cuDF are installed via the package manager.

Example:
```bash
# Launch a devcontainer that only mounts cudf and installs dependencies via conda
$ .devcontainer/launch.sh single conda

# or installs dependencies via pip
$ .devcontainer/launch.sh single pip
```

#### Unified mode

`.devcontainer/launch.sh unified` launches the devcontainer with the [default bind mounts](#host-bind-mounts), as well as additional `rmm` and `cudf` bind mounts (assumes RMM and cuDF are siblings to the cudf repository):

* `${repo}/../.rmm:/home/coder/rmm`
* `${repo}/../.cudf:/home/coder/cudf`

In this mode, RMM and cuDF will not be installed, but the devcontainer will install the dependencies necessary to build all three.

Example:
```bash
# Launch a devcontainer that mounts rmm, cudf, and cudf from the host and installs dependencies via conda
$ .devcontainer/launch.sh unified conda

# or installs dependencies via pip
$ .devcontainer/launch.sh unified pip
```

#### Isolated mode

`.devcontainer/launch.sh isolated` launches the devcontainer without the deps/repo bind mounts, and instead contains a unique copy of the `cudf` source in a Docker [volume](https://docs.docker.com/storage/volumes/).

Use this mode to launch multiple isolated development containers that can be checked out to separate branches of `cudf`.

The Docker volume persists after the devcontainer is removed, ensuring you don't pending lose work by accidentally removing the devcontainer.

However, you will need to manually remove the volume once you've committed and pushed your changes:

* Use the [`docker volume ls`](https://docs.docker.com/engine/reference/commandline/volume_ls/) command to list all volumes
* Use [`docker volume rm`](https://docs.docker.com/engine/reference/commandline/volume_rm/) or [`docker volume prune`](https://docs.docker.com/engine/reference/commandline/volume_prune/) to clean up unused volumes

Alternatively, use the "Dev Volumes" tab of the VSCode Dev Containers extension to view and remove unused volumes.

Examples:
```bash
# Launch a devcontainer that is isolated from changes on the host and installs dependencies via conda
$ .devcontainer/launch.sh isolated conda

# or installs dependencies via pip
$ .devcontainer/launch.sh isolated pip
```
