#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    pip install -r ./requirements.txt
    python video_detection.py

elif [[ "$(uname -s)" == "Darwin" ]]; then
    python3 -m pip install -r ./requirements.txt
    python3 video_detection.py

elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    pip install -r .\requirements.txt
    python video_detection.py

else
    echo "Unsupported operating system."
    exit 1
fi