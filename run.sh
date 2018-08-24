#!/usr/bin/env sh

pyenv local 3.6.4
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python -m transcode
python -m synth.prepare
python -m synth.train
