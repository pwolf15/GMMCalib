#!/bin/bash

RERUN=ON

# cd to repo root
cd "$(dirname "$0")/.."

# cad calib
python -m gmmcalib.calibrate \
 --data_path ../data/single_chair/ \
 --config_file_path ../config/config_single_chair.yaml \
 --model_path ../data/models/chair.obj \
 --method cad_calib 