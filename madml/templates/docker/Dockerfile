# Initial setup
FROM ubuntu

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home

RUN apt update
RUN apt upgrade -y

RUN apt install python3-pip git -y
RUN pip install -U pip
RUN git clone https://github.com/leschultz/materials_application_domain_machine_learning.git
RUN cd materials_application_domain_machine_learning && pip install -e .

COPY model.dill .

COPY model_predict.py .
