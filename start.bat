@echo off

@REM start "" stt_service.exe

start "" cmd /k "conda activate chat && python modules\llm_server.py"

start "" cmd /k "conda activate chat && python modules\emoji.py"

start "" cmd /k "conda activate chat && python main.py"

start "" cmd /k "cd blivechat && blivechat.exe"

echo 已启动所有服务