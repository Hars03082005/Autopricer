FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY model_artifacts ./model_artifacts

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9000"]