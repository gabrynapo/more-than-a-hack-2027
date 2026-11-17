@echo off
rem Windows version of scripts/run_local_v11.sh: generate dfl_sup_v11 (v10 with the four extra
rem feature families, every model retrained on them: docs/MIGLIORAMENTI_SPELLF1.md section 34.3)
rem twice, check that both runs give the same file, validate it and score it on the public 5 minutes.
rem Run it yourself, locally, from the repository folder: it reads the restricted data in data\.
rem Prints aggregates only.
rem
rem     scripts\run_local_v11.bat
rem
rem Needs Python 3.11 (the "py" launcher). Adopt v11 only if: REPRODUCIBLE, whole-match D share in
rem 40-50%%, 2-4.5 possessions per minute, public-window dead-ball recall >= 0.66. Else submit v10.
setlocal
cd /d "%~dp0.."
set PY=.venv\Scripts\python.exe
set ARGS=--networks light_symsw --evidence-weight 0.25 --extra-features %MODELS%
if not exist %PY% py -3.11 -m venv .venv || goto :error
%PY% -m pip install -q -r requirements-pipeline.txt || goto :error
if "%MODELS%"=="" %PY% scripts\fetch_dfl.py || goto :error
%PY% scripts\make_submission.py --name dfl_sup_v11 %ARGS% || goto :error
%PY% scripts\make_submission.py --name dfl_sup_v11_rerun %ARGS% || goto :error
fc /b evaluation\results\dfl_sup_v11.csv evaluation\results\dfl_sup_v11_rerun.csv > nul && (echo REPRODUCIBLE: identical files) || (echo NOT REPRODUCIBLE: files differ)
del evaluation\results\dfl_sup_v11_rerun.csv
certutil -hashfile evaluation\results\dfl_sup_v11.csv SHA256
%PY% evaluation\validate_submission.py evaluation\results\dfl_sup_v11.csv || goto :error
%PY% evaluation\run_evaluation.py
goto :eof
:error
echo FAILED at the step above
exit /b 1
