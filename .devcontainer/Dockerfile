# syntax=docker/dockerfile:1.5

ARG PIP_BASE
ARG CONDA_BASE

ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${PIP_BASE} as pip-base

FROM ${CONDA_BASE} as conda-base

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
