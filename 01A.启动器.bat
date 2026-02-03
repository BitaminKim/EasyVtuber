@echo on
cd /D "%~dp0"

SET PATH=%~dp0envs\TensorRT-RTX-1.3.0.35_cu129\bin;%~dp0envs\python_embedded;%~dp0envs\python_embedded\Scripts;%~dp0envs\python_embedded\Library\bin;%PATH%

pythonw launcher2.py