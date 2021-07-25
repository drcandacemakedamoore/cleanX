"%PYTHON%" setup.py bdist_egg
if errorlevel 1 exit 1
for /f "tokens=* USEBACKQ" %%a in (`"%PYTHON%" setup.py find_egg`) do @set egg=%%a
if errorlevel 1 exit 1
"%PYTHON%" setup.py easy_install --record=record.txt --no-deps %egg%
if errorlevel 1 exit 1
