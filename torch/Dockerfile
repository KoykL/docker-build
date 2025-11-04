FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# Install dependencies for OpenBLAS, Jupyter, and Torch
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y \
    build-essential git gfortran \
    python3 python3-setuptools python3-dev \
    cmake curl wget unzip libreadline-dev libjpeg-dev libpng-dev ncurses-dev \
    imagemagick gnuplot gnuplot-x11 libssl-dev libzmq3-dev graphviz libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Torch
RUN git clone https://github.com/nagadomi/distro.git ~/torch --recursive && \
    cd ~/torch && \
    ./install.sh
ENV LUA_PATH='/root/.luarocks/share/lua/5.1/?.lua;/root/.luarocks/share/lua/5.1/?/init.lua;/root/torch/install/share/lua/5.1/?.lua;/root/torch/install/share/lua/5.1/?/init.lua;./?.lua;/root/torch/install/share/luajit-2.1.0-alpha/?.lua;/usr/local/share/lua/5.1/?.lua;/usr/local/share/lua/5.1/?/init.lua' \
    LUA_CPATH='/root/.luarocks/lib/lua/5.1/?.so;/root/torch/install/lib/lua/5.1/?.so;./?.so;/usr/local/lib/lua/5.1/?.so;/usr/local/lib/lua/5.1/loadall.so' \
    PATH=/root/torch/install/bin:$PATH \
    LD_LIBRARY_PATH=/root/torch/install/lib:$LD_LIBRARY_PATH \
    DYLD_LIBRARY_PATH=/root/torch/install/lib:$DYLD_LIBRARY_PATH
COPY luarocks_config.lua /config.lua
COPY atomic.patch /
COPY shfl.patch /
ENV TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
RUN cp /config.lua ~/torch/install/etc/luarocks/config.lua && \
    cd ~/torch/extra/cutorch && \
    luarocks make rocks/cutorch-scm-1.rockspec && \
    cd ~/torch/extra/cunn && \
    luarocks make rocks/cunn-scm-1.rockspec && \
    cd .. && \
    git clone https://github.com/soumith/cudnn.torch.git -b R7 && \
    cd cudnn.torch && \
    luarocks make cudnn-scm-1.rockspec && \
    rm /config.lua && \
    rm /atomic.patch && \
    rm /shfl.patch

WORKDIR /root/

#patch -up1 < /atomic.patch && \
#patch -up1 < /shfl.patch && \
