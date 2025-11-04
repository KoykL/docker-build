FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
  git \
  cmake \
  build-essential \
  libboost-program-options-dev \
  libboost-filesystem-dev \
  libboost-graph-dev \
  libboost-system-dev \
  libboost-test-dev \
  libsuitesparse-dev \
  libfreeimage-dev \
  libmetis-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libglew-dev \
  qtbase5-dev \
  libqt5opengl5-dev \
  libcgal-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  libhdf5-dev \
  python-dev \
  python-setuptools \
  python3-pip \
  python3-dev \
  libflann-dev \
  libsqlite3-dev \
    && \
  rm -rf /var/lib/apt/lists/* && \
  apt-get clean && rm -rf /tmp/* /var/tmp/*

# Eigen
WORKDIR /opt
RUN git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git
RUN cd eigen && mkdir build && cd build && cmake .. && make install

# Ceres solver
WORKDIR /opt
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
WORKDIR /opt/ceres-solver
RUN git checkout 2.1.0rc2
RUN mkdir build
WORKDIR /opt/ceres-solver/build
RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF 
RUN make -j
RUN make install

# Colmap
WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git
WORKDIR /opt/colmap
RUN git checkout 3.8
RUN mkdir build
WORKDIR /opt/colmap/build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES=70 ..
RUN make -j
RUN make install

# PixSfM
WORKDIR /
RUN git clone https://github.com/cvg/pixel-perfect-sfm --recursive
WORKDIR /pixel-perfect-sfm
RUN pip3 install -r requirements.txt
RUN git clone --branch v1.4 --recursive https://github.com/cvg/Hierarchical-Localization/ 
WORKDIR /pixel-perfect-sfm/Hierarchical-Localization
RUN pip3 install -e .
WORKDIR /pixel-perfect-sfm
RUN pip3 install -e .
COPY run_hloc.py /pixel-perfect-sfm/

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
  less # dev tools

RUN pip install scipy