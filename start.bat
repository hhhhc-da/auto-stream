@echo off

start "" stt_service.exe

start "" cmd /k "conda activate chat && python modules\llm_server.py"

start "" blivechat\blivechat.exe

echo 已启动所有服务