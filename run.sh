#!/usr/bin/env bash
set -e

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

pyenv install -s 3.6.4
pyenv local 3.6.4
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt >/dev/null

# Clean up
rm -rf cache
rm -rf logs

python -m transcode
python -m synth.prepare
python -m synth.train
