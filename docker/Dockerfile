# syntax=docker/dockerfile:1

from python:3.9

run mkdir /src
copy . /src
run apt-get update && \
    apt-get install -qy --no-install-recommends libtesseract-dev
run cd /src && \
    python -m venv env && \
    /src/env/bin/python -m pip install wheel && \
    /src/env/bin/python setup.py install_dev

from python:3.9
run apt-get update && \
    apt-get install -qy --no-install-recommends libtesseract-dev \
    libtesseract4 \
    tesseract-ocr-all \
    libgl1
copy --from=0 /src/env/lib/python3.9/site-packages/. /usr/local/lib/python3.9/site-packages
run mkdir /cleanx
workdir /cleanx
entrypoint ["python3.9", "-m", "cleanX"]
