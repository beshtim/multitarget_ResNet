# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:22.11-py3 

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install -U pip

ARG PROJECT=multirn
ARG PROJECT_DIR=/${PROJECT}
RUN mkdir -p $PROJECT_DIR
WORKDIR $PROJECT_DIR

COPY . $PROJECT_DIR

RUN pip install tabulate xlsxwriter pandas numpy pillow kornia scikit-learn pycuda
