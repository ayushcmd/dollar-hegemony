FROM python:3.11-slim

WORKDIR /app

COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

COPY src/ ./src/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY models/ ./models/

EXPOSE 7860

CMD ["python", "src/dashboard.py"]