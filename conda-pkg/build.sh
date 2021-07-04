#!sh -xe
$PYTHON setup.py bdist_egg
$PYTHON -m easy_install --record=record.txt --no-deps ./dist/*.egg
