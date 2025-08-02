# Gunakan image Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy file requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project ke container
COPY . .

# Expose port Railway
EXPOSE 6734

# Jalankan aplikasi
CMD ["python", "app.py"]
