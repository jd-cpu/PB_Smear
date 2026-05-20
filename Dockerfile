# 1. 기반 이미지
FROM python:3.11-slim

# 2. 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# (나머지는 기존과 동일)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .