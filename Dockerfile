FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies — only reinstalls when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the rembg model weights into the image
RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

# App code — last, so only this layer rebuilds on code changes
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
