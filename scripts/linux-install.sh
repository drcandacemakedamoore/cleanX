#!/bin/sh

set -ex

if [ -z "$CONDA_DEFAULT_ENV" ] ; then
    if cat /etc/*elease | grep -i debian 2>&1 >/dev/null ; then
        sudo apt-get install -y --no-install-recommends \
             libleptonica-dev \
             tesseract-ocr-all \
             libtesseract-dev \
             python3-opencv \
             python3-venv \
             python3-pip
    else
        echo >&2 "We don't support this distribution yet. Sorry"
        exit
    fi
    if [ -z "$VIRTUAL_ENV" ] ; then
        read -p "No virtual environment detected, create one? [Y/n]?" answer
        case $answer in
            [Yy]* ) python3 -m venv .venv
                    . ./.venv/bin/activate
                    ;;
            * ) echo >&2 "Installing package system-wide (not recommended)"
                ;;
        esac
    fi
    pip install --force-reinstall --upgrade cleanX
else
    conda install -y -c conda-forge -c doctormakeda cleanx
fi

