FROM python:3.9.13-buster
#FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

COPY . /app

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt


WORKDIR /app

# change dir to /workspace

