FROM vllm/vllm-openai:v0.7.3

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies first for better layer caching.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source.
COPY llm_service /app/llm_service
COPY pyproject.toml /app/pyproject.toml

EXPOSE 8000

# Use env vars (or --env-file .env) to configure the service.
ENTRYPOINT ["uvicorn"]
CMD ["llm_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
