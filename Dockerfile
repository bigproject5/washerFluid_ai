FROM python:3.11-slim

# 영상 디코딩 및 OpenCV 런타임 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]