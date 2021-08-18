FROM ubuntu:xenial
RUN apt-get update
RUN apt-get install -y software-properties-common python-software-properties
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3.6-venv python3-pip

ADD requirements.txt /opt/ethgasstation/requirements.txt
RUN python3.6 -m pip install -r /opt/ethgasstation/requirements.txt

ADD settings.docker.conf /etc/ethgasstation.conf
ADD . /opt/ethgasstation/
ADD ethgasstation.py /opt/ethgasstation/ethgasstation.py

CMD /usr/bin/python3.6 /opt/ethgasstation/ethgasstation.py
