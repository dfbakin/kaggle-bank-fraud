FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends build-essential curl 

RUN pip install pipx \
    && python3 -m pipx ensurepath \
    && pipx install poetry

ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml /app/
RUN poetry install --no-interaction --no-ansi --no-root

RUN rm -rf ./app && mkdir -p /app/input /app/output

COPY src /app/src

CMD ["poetry", "run", "python", "src/inference.py"]
