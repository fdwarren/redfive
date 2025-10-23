FROM python:3.13-slim

WORKDIR /app/redfive
RUN pip install uv

COPY pyproject.toml .
RUN uv pip install --system -e .

COPY . .

# Add src to Python path
ENV PYTHONPATH=/app/redfive/src
ENV PORT=8080

# main.py lives inside src/
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
