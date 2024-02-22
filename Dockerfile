FROM python:3.11-slim
LABEL authors="lucas"

ENTRYPOINT ["top", "-b"]