#!/bin/sh

# Install the virtual environment tool
pip3 install virtualenv

# Create a new virtual environment
virtualenv --python=python3 budget-rnn-env

# Activate the virtual environment
. budget-rnn-env/bin/activate

# Install the necessary tools
python3 -m pip install .
