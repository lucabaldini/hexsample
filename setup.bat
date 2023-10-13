@ECHO OFF
REM See this stackoverflow question
REM http:\\stackoverflow.com\questions\3827567\how-to-get-the-path-of-the-batch-script-in-windows
REM for the magic in this command
SET WIN_SETUP_DIR=%~dp0
SET SETUP_DIR=%WIN_SETUP_DIR:\=/%

REM
REM Base package root. All the other relevant folders are relative to this
REM location.
REM
SET HEXSAMPLE_ROOT=%SETUP_DIR:~0,-1%

REM
REM Add the root folder to the $PYTHONPATH so that we can effectively import
REM the relevant modules.
REM
set PYTHONPATH=%HEXSAMPLE_ROOT%;%PYTHONPATH%

REM
REM Add the bin folder to the PATH environmental variable.
REM
set PATH=%HEXSAMPLE_ROOT%\hexsample\bin;%PATH%
