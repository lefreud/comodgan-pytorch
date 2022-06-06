#!/bin/bash

source /gel/usr/frfoc1/comodgan-styler-venv/bin/activate
pip install --quiet -r requirements.txt

# Install pycocotools
cd dataset_helpers/cocoapi/PythonAPI
make
python setup.py build_ext install
cd -
