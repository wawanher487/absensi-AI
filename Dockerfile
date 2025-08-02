# Gunakan image Python
FROM python:3.10-slim

# Install dependencies sistem yang diperlukan untuk build dlib
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja di container
WORKDIR /app

# Salin requirements.txt
COPY requirements.txt .

# Upgrade pip dan install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file project
COPY . .

# Jalankan aplikasi
CMD ["python", "app.py"]
