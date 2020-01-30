#use latest pytorch image
FROM pytorch/pytorch

ENV DEBIAN_FRONTEND noninteractive

LABEL maintainer="landon_chambers@dell.com"

#install jdk 1.8 on ubuntu and install MineRL
RUN apt-get update -y
RUN apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:openjdk-r/ppa
RUN apt-get install -y openjdk-8-jdk
RUN pip install minerl

#install plot tools and open ai gym dependencies
RUN pip install -U matplotlib
RUN pip install pandas
RUN pip install gym[atari]
RUN pip install gym-minigrid

#install xorg and xvfb for rendering in headless server, install x11vnc to view rendering.
RUN apt-get install -y xorg openbox
RUN apt-get install -y xvfb
RUN apt-get install -y git x11vnc

#Set Environment Variables
ENV DISPLAY=:20
ENV MINERL_DATA_ROOT="/workspace/data"

#Expose port 1337
EXPOSE 1337