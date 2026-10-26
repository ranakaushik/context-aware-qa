# Container image for hosts that run Docker (Hugging Face Spaces, Koyeb, Fly.io, a VM). See DEPLOYMENT.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY qa_assistant ./qa_assistant
COPY knowledge_base ./knowledge_base
COPY static ./static

# Run as a non-root user.
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn qa_assistant.api:app --host 0.0.0.0 --port ${PORT}"]
