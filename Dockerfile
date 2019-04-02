from balenalib/raspberrypi3-debian-python

# enable container init system.
ENV INITSYSTEM on

RUN apt-get update && \
    apt-get dist-upgrade && \
    apt-get install -yq --no-install-recommends \
        ffmpeg

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /usr/src/app
WORKDIR /usr/src/app

CMD ["bash","rundemo.sh"]
