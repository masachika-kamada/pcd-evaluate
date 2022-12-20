FROM mcr.microsoft.com/devcontainers/python:0-3.9
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
        meshlab \
        pcl-tools \
        x11-xserver-utils \
        xorg \
        x11-apps \
        libglu1-mesa-dev \
        mesa-utils
RUN mkdir /workspace