#!/usr/bin/env bash
set -e

# Clean up
rm -rf cache
rm -rf logs

source ./env.sh

py.test --junitxml=result.xml test

python -m transcode
python -m synth.prepare
python -m synth.train
