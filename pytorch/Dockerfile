FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
    && apt-get install -y python3 python3-dev python3-pip python3-virtualenv 

RUN virtualenv /env
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
RUN mkdir /root/.cache

RUN pip3 install --no-cache-dir torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
 