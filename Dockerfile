FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN apt-get update 

# Install essential
RUN apt-get install -y --no-install-recommends \
    libgl1-mesa-dev \
    libopenmpi-dev \
    git wget \
    python3 python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    curl ca-certificates

# Set CUDA & CUDNN path
RUN echo "export PATH=/usr/local/cuda/bin:$PATH" >> /etc/bash.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /etc/bash.bashrc

# Install UV
ADD https://astral.sh/uv/0.6.9/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory and copy file
WORKDIR /home/
COPY . /home/

# Install dependency of this repository
RUN uv venv gai-zoo --python python3.12
RUN source gai-zoo/bin/activate
RUN uv pip install .