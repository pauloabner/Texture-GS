FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Pacotes do sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    build-essential \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Variáveis para H100
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TCNN_CUDA_ARCHITECTURES=90
ENV TORCH_CUDA_ARCH_LIST="9.0"
ENV FORCE_CUDA=1
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs

# Correção definitiva para o linker encontrar o CUDA durante o build
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so

# Instalação PyTorch
RUN python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN python -m pip install --no-cache-dir ninja fvcore iopath

WORKDIR /app
COPY . .

# Instalação das bibliotecas que exigem compilação
RUN python -m pip install --no-cache-dir -r requirements.txt --no-build-isolation
# RUN python -m pip install --no-cache-dir tiny-cuda-nn/bindings/torch/. --no-build-isolation
RUN python -m pip install --no-cache-dir nvdiffrast/. --no-build-isolation
RUN python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"