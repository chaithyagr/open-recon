# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev


# ISMRMRD library
RUN mkdir -p /opt/code && cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# ----- 2. Build Python environment (with CUDA compiler available) -----
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel AS python-mrd-builder

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git build-essential libxslt1.1 libhdf5-103 libboost-program-options1.74.0 libpugixml1v5 dos2unix

# Install Python packages that require CUDA to compile
RUN pip3 install mri-nufft gpuNUFFT cufinufft finufft cupy-cuda11x \
                 finufft scikit-image joblib git+https://github.com/mind-inria/ggrappa@openrecon nvitop && \
    pip3 install matplotlib==3.8.2 pydicom==3.0.1 h5py==3.10.0 ismrmrd==1.14.1

# Install ISMRMRD Python tools
RUN mkdir -p /opt/code && cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd ismrmrd-python-tools && pip3 install --no-cache-dir .

# Optional cleanup
RUN rm -rf /root/.cache/pip

# ----- 3. Final runtime image -----
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime AS python-mrd-runtime

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"
LABEL org.opencontainers.image.url="https://github.com/kspaceKelvin/python-ismrmrd-server"
LABEL org.opencontainers.image.authors="Kelvin Chow (kelvin.chow@siemens-healthineers.com)"

# Copy ISMRMRD and siemens_to_ismrmrd binaries from stage 1
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd /usr/local/bin/siemens_to_ismrmrd

# Runtime dependencies only
RUN apt-get update && apt-get install --no-install-recommends -y \
    libxslt1.1 libhdf5-103 libboost-program-options1.74.0 libpugixml1v5 git dos2unix nano htop && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all built Python packages from builder
COPY --from=python-mrd-builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# Set up workspace
RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

RUN if [ -d /opt/code/python-ismrmrd-server/python-ismrmrd-server ]; then \
    echo "docker build should be run inside of python-ismrmrd-server instead of one directory up"; exit 1; fi

RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix && \
    find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \; && \
    echo "set positionlog" > ~/.nanorc

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/code/python-ismrmrd-server
CMD ["/bin/bash", "-c", "/usr/sbin/ldconfig && exec python3 main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=sparkling"]

