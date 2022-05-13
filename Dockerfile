FROM python:3.7

WORKDIR /antigenclassifier

COPY Makefile Makefile

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

CMD ["python", "serving.py"]