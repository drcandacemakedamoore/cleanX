#!sh -xe
$PYTHON setup.py bdist_egg
$PYTHON setup.py easy_install --record=record.txt --no-deps ./dist/*.egg
