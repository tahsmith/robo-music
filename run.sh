#!/usr/bin/env bash
set -e
pyenv local 3.6.4
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Clean up
rm -rf cache
rm -rf logs

python -m transcode
python -m synth.prepare
python -m synth.train
