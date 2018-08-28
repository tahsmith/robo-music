#!/usr/bin/env bash

PY_VERSION=3.6.4

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi

pyenv install -s ${PY_VERSION}
pyenv local ${PY_VERSION}
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt >/dev/null