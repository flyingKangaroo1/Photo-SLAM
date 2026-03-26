# 1. 베이스 이미지 설정 (호스트 CUDA 12.9와 호환되는 11.8 devel 사용)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 기본 도구 및 GCC 11 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    sudo \
    curl \
    zip \
    unzip \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get install -y gcc-11 g++-11 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# 3. Photo-SLAM 및 ORB-SLAM3 시스템 의존성 설치
RUN apt-get install -y \
    libeigen3-dev \
    libboost-all-dev \
    libjsoncpp-dev \
    libopengl-dev \
    mesa-utils \
    libglfw3-dev \
    libglm-dev \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev \
    libnvtoolsext1 \
    && rm -rf /var/lib/apt/lists/*

# 4. CMake 3.22.1 설치
RUN wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-Linux-x86_64.sh -O /cmake.sh && \
    chmod +x /cmake.sh && /cmake.sh --skip-license --prefix=/usr/local

# 5. OpenCV 4.8.0 빌드 (CUDA 및 Contrib 포함)
RUN mkdir /opencv && cd /opencv && \
    wget https://github.com/opencv/opencv/archive/refs/tags/4.8.0.zip -O opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.0.zip -O contrib.zip && \
    unzip opencv.zip && unzip contrib.zip && \
    mkdir -p opencv-4.8.0/build && cd opencv-4.8.0/build && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE \
          -DWITH_CUDA=ON \
          -DWITH_CUDNN=ON \
          -DOPENCV_DNN_CUDA=ON \
          -DWITH_NVCUVID=ON \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          -DOPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib-4.8.0/modules \
          -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON .. && \
    make -j$(nproc) && make install && ldconfig

# 6. LibTorch 2.4.0 (Nightly, cu118) 설치
RUN wget https://download.pytorch.org/libtorch/nightly/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0.dev20240425%2Bcu118.zip -O /libtorch.zip && \
    unzip /libtorch.zip -d / && rm /libtorch.zip

ENV Torch_DIR /libtorch/share/cmake/Torch

# 7. 시각화 및 평가를 위한 Python 패키지 설치
RUN pip3 install numpy scipy scikit-image lpips pillow tqdm plyfile evo


