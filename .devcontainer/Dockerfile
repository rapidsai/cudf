# syntax=docker/dockerfile:1.5

ARG CUDA=12.0
ARG LLVM=16
ARG RAPIDS=23.10
ARG DISTRO=ubuntu22.04
ARG REPO=rapidsai/devcontainers

ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${REPO}:${RAPIDS}-cpp-llvm${LLVM}-cuda${CUDA}-${DISTRO} as pip-base

FROM ${REPO}:${RAPIDS}-cpp-mambaforge-${DISTRO} as conda-base

COPY --from=pip-base /etc/skel/.config/clangd/config.yaml /etc/skel/.config/clangd/config.yaml

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"
ENV HISTFILE="/home/coder/.cache/._bash_history"
