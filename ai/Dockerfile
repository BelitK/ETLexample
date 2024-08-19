FROM python:3.9

COPY . .
RUN apt-get update
RUN apt-get install default-jdk -y
RUN pip install --no-cache-dir --upgrade -r req.txt

# If running behind a proxy like Nginx or Traefik add --proxy-headers

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "2000"]