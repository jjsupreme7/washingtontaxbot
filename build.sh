#!/bin/bash

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data to a known location
python -m nltk.downloader punkt stopwords -d /opt/render/nltk_data
