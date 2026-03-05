FROM python:3.12-slim

# System deps for EasyOCR (OpenCV, libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY detection_layer/ detection_layer/
COPY backend/ backend/

# Download ML models at build time
RUN cd detection_layer && \
    python download_model.py && \
    python download_layer2_model.py

# HF Spaces expects port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "backend"]
