FROM python:3.10.8
EXPOSE 80
COPY requirements.txt /opt
RUN pip install -r opt/requirements.txt
RUN apt-get update -q -y && apt-get install -q -y libgl1
COPY . /opt
WORKDIR opt
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80", "--workers","4"]
