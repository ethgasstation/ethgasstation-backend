FROM ubuntu:bionic
RUN apt-get update
RUN apt-get install -y python3 python3-pip git netcat

RUN git clone https://github.com/eficode/wait-for.git

RUN cp wait-for/wait-for /opt/

ADD requirements.txt /opt/ethgasstation/requirements.txt
RUN pip3 install -r /opt/ethgasstation/requirements.txt

ADD settings.docker.conf /etc/ethgasstation.conf
ADD . /opt/ethgasstation/
ADD ethgasstation.py /opt/ethgasstation/ethgasstation.py

CMD /opt/wait-for mariadb:3306 -- /usr/bin/python3 /opt/ethgasstation/ethgasstation.py
