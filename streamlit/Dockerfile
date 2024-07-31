FROM python:3.9

COPY . .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# If running behind a proxy like Nginx or Traefik add --proxy-headers

CMD ["streamlit", "run", "app.py"]