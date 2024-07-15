#!/bin/bash

# Create and activate the conda environment
conda env create -f environment.yml
conda activate facial_recognition

# Run the setup script to download Haar Cascade files
python setup.py
