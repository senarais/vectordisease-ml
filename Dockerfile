# Gunakan image Python ya# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements dulu untuk memancing Docker caching layer
COPY requirements.txt .

# Install semua library tanpa menyimpan cache biar image tetap kecil
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua sisa kode (main.py, ridge_model.pkl) ke dalam /app
COPY . .

# Expose port yang baru (8001)
EXPOSE 8001

# Perintah untuk menjalankan server di port 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
