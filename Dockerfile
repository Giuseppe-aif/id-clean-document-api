FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from rembg import new_session; new_session('isnet-general-use')"

# cache-bust 2025-06-08-v9
ADD https://raw.githubusercontent.com/Giuseppe-aif/id-clean-document-api/main/main.py?v=9 /app/main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
