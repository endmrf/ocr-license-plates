FROM python:3.12

COPY . .

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]