FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /backend

COPY src .

COPY requirement.txt .

WORKDIR /backend/src/detectron2

RUN pip install torch

RUN pip install -e .

WORKDIR /backend

RUN pip install -r requirement.txt

CMD ["uvicorn", "main:app", "--reload"]
