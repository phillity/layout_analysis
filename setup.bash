#!/bin/bash

git clone https://huggingface.co/SenseTime/deformable-detr
mkdir model
mv deformable-detr model/deformable-detr

wget https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip
unzip DocLayNet_core.zip -d DocLayNet
