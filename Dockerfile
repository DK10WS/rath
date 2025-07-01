FROM ghcr.io/astral-sh/uv:python3.9-bookworm

WORKDIR /backend

COPY src .

COPY requirement.txt .

RUN pip install torch

WORKDIR /backend/detectron2

RUN pip install -e .

WORKDIR /backend

RUN pip install -r requirement.txt

CMD ["uvicorn", "main:app", "--reload"]
