FROM python:3.9

RUN apt update
RUN apt install -y gdal-bin libgdal-dev libgl1-mesa-glx exiftool

COPY requirements.txt requirements.txt
# install numpy first to make sure we have it set up for gdal
RUN pip install numpy
RUN pip install -r requirements.txt
