FROM tensorflow/tensorflow:latest-py3
MAINTAINER vantis <lllvantis@163.com>
COPY . /app
WORKDIR /app
RUN pip install --upgrade flask matplotlib sklearn
CMD ["python3 basic_classification.py"]
