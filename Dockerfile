# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV RUNTIME=docker
ENV DEBIAN_FRONTEND=noninteractive

# Fix truly devious bug when building on MacOS ARM chip, see https://github.com/docker/for-mac/issues/7025#issuecomment-1755988838
ARG BUILDPLATFORM
RUN if [ "$BUILDPLATFORM" = "linux/arm64" ]; then \
    echo 'Acquire::http::Pipeline-Depth "0";\nAcquire::http::No-Cache "true";\nAcquire::BrokenProxy "true";\n' > /etc/apt/apt.conf.d/99fixbadproxy; \
    fi

# install some basics, 
# micro for having an actually good terminal-based editor
# curl for getting uv, bc for gpu scheduling util script 
RUN apt-get update && apt-get install --fix-missing -y git gcc g++ nano openssh-client curl bc && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

# Create a non-privileged user
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
ARG USERHOME="/home/nlpresearcher"
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home $USERHOME \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    nlpresearcher

# make important dirs with correct permissions for non-priviliged user
# use chmod -777 instead of chown s.t. docker run with different UID also works
RUN mkdir /opt/venv && chmod -R 777 /opt/venv/ && \
    mkdir /opt/reqs && chmod -R 777 /opt/reqs/ && \
    mkdir -p $USERHOME/.cache/uv && chmod -R 777 $USERHOME/.cache/uv && \
    mkdir -p $USERHOME/.local/bin && chmod -R 777 $USERHOME/.local/bin && \
    mkdir -p $USERHOME && chmod -R 777 $USERHOME 

# good to copy these instead of mounting, that way the requirements can be easily reconstructed just from having the image 
COPY ./uv.lock /opt/venv/uv.lock
COPY ./pyproject.toml /opt/venv/pyproject.toml
COPY ./.python-version /opt/venv/.python-version

# Switch to non-root user now to prevent issues when running w/ non-root uid later on
USER nlpresearcher
WORKDIR $USERHOME
# Install uv
ADD --chmod=755 https://astral.sh/uv/0.6.5/install.sh ./install.sh
RUN ./install.sh && rm -rf ./install.sh 
# ...and add uv to PATH
ENV PATH="${USERHOME}/.local/bin:${PATH}"

# switch to uv "project"
WORKDIR /opt/venv/

# CC / CXX env vars needed for fasttext / flash-attn
ENV CC="/usr/bin/gcc"
ENV CXX="/usr/bin/g++"

# install from lockfile, mount cache
# b/c of flash-attn, we do 2-step process, see https://github.com/astral-sh/uv/issues/6437#issuecomment-2535324784
RUN --mount=type=cache,target=$USERHOME/.cache/uv,uid=$UID uv sync --frozen --no-install-package flash-attn && uv sync --frozen

# ``activate'' venv automatically when using docker run
ENV PATH="/opt/venv/.venv/bin:${PATH}"

# set this as default entrypoint workdir, code should be mounted there
WORKDIR /workspace