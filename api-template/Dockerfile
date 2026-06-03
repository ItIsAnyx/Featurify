FROM python:3.12-slim

WORKDIR /code

# System deps kept minimal; httpx + sqlmodel are pure-python wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Directory for the SQLite database + uploaded datasets (mounted as a volume)
RUN mkdir -p /data/datasets

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
