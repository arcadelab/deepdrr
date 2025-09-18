mkdir -p /usr/share/glvnd/egl_vendor.d/

echo "{\n\
\"file_format_version\" : \"1.0.0\",\n\
\"ICD\": {\n\
\"library_path\": \"libEGL_nvidia.so.0\"\n\
}\n\
}" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

apt-get update

# Should be the same list of packages as in the Dockerfile, make sure to keep them in sync
apt-get install \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    pkg-config \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    freeglut3-dev
