# Gunakan base image Python slim
FROM python:3.10-slim

# Ubah ke non-interaktif agar tidak hang saat build
ENV DEBIAN_FRONTEND=noninteractive

# Ganti mirror APT agar lebih cepat dan stabil (opsional)
RUN sed -i 's|http://deb.debian.org|http://ftp.de.debian.org|g' /etc/apt/sources.list

# Install dependencies yang dibutuhkan secara bertahap
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libboost-all-dev \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Salin file requirements terlebih dahulu (agar caching efisien)
COPY requirements.txt .

# Install python dependencies lebih dulu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek setelah install dependensi
COPY . .

# Jalankan aplikasi
CMD ["python", "app.py"]
