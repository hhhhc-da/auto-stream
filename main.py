from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import json
import requests
import pyautogui
import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosedOK
import asyncio

from modules.voice_gen import advanced_generate_voice

app = FastAPI()

origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

async def chat_deepseek(question:str='', prompt_path=os.path.join('prompts', 'prompt.txt'), encoding='utf-8'):
    '''
    WebSockets 流式访问函数
    '''
    uri = "ws://localhost:82/ws"
    async with websockets.connect(uri) as websocket:
        long_string = ""

        message = ''
        with open(prompt_path, "r", encoding=encoding) as f:
            message = f.read()
            
        message += "网友:" + question + "\n回复:"
        await websocket.send(message)

        try:
            while True:
                response = await websocket.recv()
                long_string += response
                
                if len(long_string) > 50:
                    print("数据超长")
                    raise Exception("数据超长")
        except ConnectionClosedOK:
            # print("Server closed the connection normally.")
            pass
        except Exception as e:
            websocket.close()
            print("Exception:", e)
        finally:
            return long_string

async def chat_async(question:str=''):
    '''
    异步流式访问 DeepSeek 封装函数
    '''
    return await chat_deepseek(question) 

@app.post("/text")
async def receive(request: Request):
    data = await request.json()
    text = json.loads(data)["text"]
    
    response = requests.post('http://127.0.0.1:81/predict', json=json.dumps({"text": text}))
    if response.status_code != 200:
        return {"error": "Failed to send text to prediction service"}
    
    emoji = response.json()["type"]
    if emoji != 'daily':
        print(f"(Log) Received emoji: {emoji}, press SHIFT + F.")
        pyautogui.press(['shift', 'f'])
    else:
        print("(Log) No emoji detected, no action taken.")
    
    # 使用 DeepSeek 自动回复
    try:
        response = await chat_async(question=text)
    except Exception as e:
        print(f"DeepSeekChatError: {e}")
        
    try:
        if os.path.exists("audio") is False:
            os.makedirs("audio")    
        if len(os.listdir("audio")) > 0:
            for file in os.listdir("audio"):
                os.remove(os.path.join("audio", file))
        
        wavs = advanced_generate_voice([response], temperature=0.3, top_P=0.7, top_K=20, prompt='[laugh_1]')
        sd.play(wavs[0], samplerate=24000)

    except Exception as e:
        print(f"语音生成失败: {e}")

    return json.dumps({"status": "success", "message": "Text received and processed"})

if __name__ == "__main__":
    # 启动 FastAPI 服务
    uvicorn.run('main:app', host='0.0.0.0', port=83)