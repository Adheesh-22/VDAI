FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

ARG MODEL_PATH=model.onnx
ARG HOST=0.0.0.0
ARG PORT=8000
ARG LOG_LEVEL=INFO

ENV MODEL_PATH=${MODEL_PATH}
ENV HOST=${HOST}
ENV PORT=${PORT}
ENV LOG_LEVEL=${LOG_LEVEL}

CMD ["python", "main.py"]
