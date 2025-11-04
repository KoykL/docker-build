FROM ubuntu:22.04 AS builder
# RUN dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm && dnf groupinstall -y "Development Tools" && dnf install -y python3.11 python3.11-devel python3.11-pip cmake ninja-build libatomic ccache gcc-toolset-10  && dnf clean all
RUN apt-get update && apt-get install -y python3-dev python3-pip ccache cmake ninja-build  
RUN pip install --no-cache-dir scikit-build
COPY . /mitsuba
WORKDIR /mitsuba
ENV CCACHE_DIR=/ccache
ARG NLTO=auto
RUN --mount=type=cache,target=/ccache cd ext/pybind11 && rm -rf dist && CMAKE_CXX_COMPILER_LAUNCHER=ccache python3 setup.py bdist_wheel && pip3 install ./dist/*.whl
RUN --mount=type=cache,target=/ccache cd ext/drjit && rm -rf dist  && CMAKE_CXX_COMPILER_LAUNCHER=ccache python3 setup.py bdist_wheel -- -DCMAKE_C_FLAGS=-flto=${NLTO} -DCMAKE_CXX_FLAGS=-flto=${NLTO} && pip3 install ./dist/*.whl
RUN --mount=type=cache,target=/ccache rm -rf dist && CMAKE_CXX_COMPILER_LAUNCHER=ccache python3 setup.py bdist_wheel -- -DCMAKE_C_FLAGS=-flto=${NLTO} -DCMAKE_CXX_FLAGS=-flto=${NLTO}

FROM scratch AS wheels
COPY --from=builder /mitsuba/dist/*.whl /
COPY --from=builder /mitsuba/ext/drjit/dist/*.whl /