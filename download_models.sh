#!/usr/bin/env bash

echo "Downloading config files..."

mkdir cfg
wget -O cfg/coco.data https://public-model.s3-ap-southeast-1.amazonaws.com/coco.data
wget -O cfg/yolov3-data.cfg https://public-model.s3-ap-southeast-1.amazonaws.com/yolov3-data.cfg
wget -O cfg/voc.data https://public-model.s3-ap-southeast-1.amazonaws.com/voc_v2.data

echo "Modify config parameters to enable Testing mode"
sed -i '/batch=64/c\# batch=64' cfg/yolov3-data.cfg
sed -i '/subdivisions=16/c\# subdivisions=16' cfg/yolov3-data.cfg
sed -i '/# batch=1/c\batch=1' cfg/yolov3-data.cfg
sed -i '/# subdivisions=1/c\subdivisions=1' cfg/yolov3-data.cfg

mkdir data
wget -O data/voc.names https://public-model.s3-ap-southeast-1.amazonaws.com/voc.names

echo "Downloading yolov3 weights"
mkdir weights
wget -O weights/yolov3-data_final_26.weights https://public-model.s3-ap-southeast-1.amazonaws.com/yolov3-data_final_26.weights
