FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

ENTRYPOINT ["uiregress"]
CMD ["info"]
