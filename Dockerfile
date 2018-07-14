FROM ubuntu:xenial
RUN apt-get update
RUN apt-get install -y python3 python3-pip

ADD requirements.txt /opt/ethgasstation/requirements.txt
RUN pip3 install -r /opt/ethgasstation/requirements.txt

ADD settings.docker.conf /etc/ethgasstation.conf
ADD . /opt/ethgasstation/
ADD ethgasstation.py /opt/ethgasstation/ethgasstation.py

CMD /usr/bin/python3 /opt/ethgasstation/ethgasstation.py
