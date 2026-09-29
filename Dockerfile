FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY crag crag
COPY data data
COPY web web
COPY docs docs
EXPOSE 8000
CMD ["uvicorn", "crag.api:app", "--host", "0.0.0.0", "--port", "8000"]
