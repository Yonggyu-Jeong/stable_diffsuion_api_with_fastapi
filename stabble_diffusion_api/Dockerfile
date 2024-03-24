ARG CUDA_VERSION=12.1
ARG PYTHON_VERSION=3.10

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04
FROM python:${PYTHON_VERSION}

LABEL authors="Yonggyu_Jeong"

WORKDIR /app

COPY AiLT /app

RUN apt-get update && apt-get install -y fonts-nanum

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY requirements_docker.txt .

RUN pip install fastapi uvicorn
RUN pip3 install torch torchvision torchaudio
RUN pip install --no-cache-dir -r requirements_docker.txt


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t ailt_with_ml .