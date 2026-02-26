FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install uv

# Copy project config and install dependencies
COPY pyproject.toml README.md ./
COPY backend/ backend/
RUN uv pip install --system -e "."

EXPOSE 8000

CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
