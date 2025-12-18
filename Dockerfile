FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
COPY heroprotocol /app/heroprotocol

RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core fonts-noto-cjk libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
COPY heroprotocol /app/heroprotocol
COPY app /app/app
COPY data/images /app/data/images

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.replay_parser.src.parser_api:app", "--host", "0.0.0.0", "--port", "8000"]
