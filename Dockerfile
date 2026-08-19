FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/core

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgmp-dev \
    && rm -rf /var/lib/apt/lists/*
COPY server-requirements.txt ./
COPY kademlia-lib/ ./kademlia-lib/
COPY core/ ./core/
RUN python -m pip install --upgrade pip \
    && pip install -r server-requirements.txt

CMD ["python", "-m", "core.server_main"]
