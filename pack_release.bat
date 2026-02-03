@echo on
REM ========================================================================
REM IMPORTANT: Before running this script, please make sure to:
REM 1. Checkout to the release_package branch: git checkout release_package
REM 2. Rebase with the latest changes: git rebase main
REM ========================================================================

cd /D "%~dp0"

if not exist "envs" mkdir envs
if not exist "envs\miniconda3" mkdir envs\miniconda3

IF not EXIST %~dp0envs\TensorRT-RTX-1.3.0.35_cu129\bin (
    @RD /S /Q %~dp0envs\TensorRT-RTX-1.3.0.35_cu129
    cd /D "%~dp0\envs"
    curl -L -o trt_rtx.zip https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/1.3/TensorRT-RTX-1.3.0.35-win10-amd64-cuda-12.9-Release-external.zip
    tar -xf trt_rtx.zip "TensorRT-RTX-1.3.0.35" && ren "TensorRT-RTX-1.3.0.35" "TensorRT-RTX-1.3.0.35_cu129" && del trt_rtx.zip
    cd /D "%~dp0"
)

IF not EXIST %~dp0envs\miniconda3\Scripts (
    @RD /S /Q %~dp0envs\miniconda3
    mkdir %~dp0envs\miniconda3
    echo "Downloading miniconda..."
    powershell -Command "Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-py310_24.9.2-0-Windows-x86_64.exe' -OutFile '.\envs\miniconda3.exe' -UseBasicParsing -Headers @{'User-Agent'='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'}"

    echo "Installling minconda..."
    start /wait "" %~dp0envs\miniconda3.exe /S /AddToPath=0 /RegisterPython=0 /InstallationType=JustMe /D=%~dp0envs\miniconda3
    echo "Successfully install minconda"
    del envs\miniconda3.exe
)

SET PATH=%~dp0envs\TensorRT-RTX-1.3.0.35_cu129\bin;%~dp0envs\miniconda3\Scripts;%PATH%

call activate
call conda env list
call conda update -y --all
call conda create -n ezvtb_rt_venv_release python=3.10 -y
call conda activate ezvtb_rt_venv_release
call conda env list

call conda install -y conda-pack
call conda install -y conda-forge::pycuda 
call conda install -y -c nvidia/label/cuda-12.9.1 cuda-nvcc-dev_win-64 cudnn cuda-runtime

call conda-pack -n ezvtb_rt_venv_release -o %~dp0envs\python_embedded --format no-archive


SET PATH=%~dp0envs\python_embedded;%~dp0envs\python_embedded\Scripts;%~dp0envs\python_embedded\Library\bin;%PATH%
call python -m pip install %~dp0envs\TensorRT-RTX-1.3.0.35_cu129\python\tensorrt_rtx-1.3.0.35-cp310-none-win_amd64.whl
call python -m pip install -r requirements.txt --no-warn-script-location

@RD /S /Q %~dp0envs\miniconda3
pause