@echo off

rem 启动STT服务（在新窗口中运行）
start "" stt_service.exe

rem 激活conda环境并启动Python服务（在新窗口中运行）
start "" cmd /k "conda activate chat && python modules\llm_server.py"

echo 已启动所有服务