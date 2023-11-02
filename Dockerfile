FROM nvidia/cuda:12.0.0-devel-ubuntu22.04 as pvm_base

ENV DISPLAY=:1.0
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
     vim \
     git \
     python3-pip \
     python3-opencv \
     netcat \
     telnet 
RUN pip3 install pycuda 
COPY entrypoint.sh /
ENTRYPOINT [ "/entrypoint.sh" ]
#ENTRYPOINT "/bin/bash -c '/usr/bin/pip3 install -e .' && /bin/bash"
