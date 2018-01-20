FROM ubuntu:xenial
RUN apt-get update
RUN apt-get install -y python3 python3-pip

ADD settings.docker.conf /etc/ethgasstation.conf
ADD ethgasstation.py /opt/ethgasstation/ethgasstation.py
ADD model_gas.py /opt/ethgasstation/model_gas.py
ADD requirements.txt /opt/ethgasstation/requirements.txt
ADD egs/ /opt/ethgasstation/egs/
RUN pip3 install -r /opt/ethgasstation/requirements.txt

CMD /usr/bin/python3 /opt/ethgasstation/ethgasstation.py
