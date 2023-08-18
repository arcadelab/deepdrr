# syntax=docker/dockerfile:1

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime as base
WORKDIR /app

COPY scripts/ubuntu_setup.sh .
RUN bash ubuntu_setup.sh

RUN mkdir -p /usr/share/glvnd/egl_vendor.d/ && \
    echo "{\n\
    \"file_format_version\" : \"1.0.0\",\n\
    \"ICD\": {\n\
    \"library_path\": \"libEGL_nvidia.so.0\"\n\
    }\n\
    }" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY environment.yml .
RUN conda env update -n base -f environment.yml

COPY . .
RUN pip install .[dev,cuda11x]

# CMD python tests/test_core.py
CMD python -m pytest -v
