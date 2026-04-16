#!/usr/bin/env bash

# script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
#
#  Adapted from:
#  https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
#
#  data/imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  data/imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Make imagnet directory
#
CURRENT_DIR=$(pwd)
IMAGENET_ROOT="data/imagenet"
mkdir -p "${IMAGENET_ROOT}"
#
# Extract the training data:
#
# Create train directory; move .tar file; change directory
mkdir -p "${IMAGENET_ROOT}/train" && mv ILSVRC2012_img_train.tar "${IMAGENET_ROOT}/train/" && cd "${IMAGENET_ROOT}/train" || exit
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME; do
  mkdir -p "${NAME%.tar}"
  tar -xvf "${NAME}" -C "${NAME%.tar}"
  rm -f "${NAME}"
done
#
# This results in a training directory like so:
#
#  data/imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
cd "${CURRENT_DIR}" || exit
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir -p "${IMAGENET_ROOT}/val" && mv ILSVRC2012_img_val.tar "${IMAGENET_ROOT}/val/" && cd "${IMAGENET_ROOT}/val" && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
#
#  data/imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#
