FROM python:3.10.13 AS builder
COPY ./requirements.txt /home
RUN pip install -r /home/requirements.txt

FROM python:3.10.13-slim-bullseye
EXPOSE 8000
WORKDIR /home
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . /home
ENV WATCHFILES_FORCE_POLLING=true
ENTRYPOINT ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--reload", "--reload-include", "*.yaml"]