@echo off
rem Fast variant of scripts\run_local_v11.bat: loads the committed DFL-trained models from models\
rem (the *_xf files; no DFL download, no training). Otherwise identical.
rem
rem     scripts\run_local_v11_fast.bat
setlocal
set MODELS=--model-dir models
call "%~dp0run_local_v11.bat"
