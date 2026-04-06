FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/     ./src/
COPY app/     ./app/
COPY data/    ./data/
COPY config.json ./

# Pre-train models at build time
RUN python -m src.train

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
