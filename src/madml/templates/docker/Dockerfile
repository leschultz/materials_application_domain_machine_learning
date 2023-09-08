# Initial setup
FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home

RUN apt update
RUN apt upgrade -y

RUN apt install python3-pip -y
RUN pip install -U pip
RUN pip install madml

# Copy the user's packages
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.dill .
COPY model_predict.py .

CMD ["python3", "model_predict.py"]
