#!/bin/bash

if source venv/bin/activate || source .venv/bin/activate
then
  echo "found venv, instaling libs"
else
  echo "no venv found, creating venv and instaling libs"
  if python3 --version
  then
    echo "No python found, please install python https://www.python.org/downloads/"
    exit 2
  fi
  python3 -m venv .venv
  source .venv/bin/activate
fi

pip install --requirement requirements.txt