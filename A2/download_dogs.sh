#! /bin/bash
mkdir images && cd images &&\
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar &&\
tar -xf images.tar && rm images.tar &&\
find . -mindepth 2 -type f -iname "*.jpg" -exec mv -n {} . \; &&\
find . -type d -empty -delete
