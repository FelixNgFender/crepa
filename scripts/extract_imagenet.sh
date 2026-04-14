#!/usr/bin/env bash

# script to extract ImageNet validation dataset
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_val.tar in your current directory
#
#  Adapted from:
#  https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
#
#  data/imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir -p data/imagenet/val && mv ILSVRC2012_img_val.tar data/imagenet/val/ && cd data/imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Check total files after extract
#
# find val/ -name "*.JPEG" | wc -l
# 50000
