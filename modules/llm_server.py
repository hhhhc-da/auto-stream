# coding=utf-8
from llama_cpp import Llama
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import websockets
from websockets.exceptions import ConnectionClosedOK
import asyncio
import pandas as pd
import re

app = FastAPI()
llama = Llama(model_path=os.path.abspath(os.path.join("models", "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")))

origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 全局标志位，用于控制线程是否继续运行
running = True

@app.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    '''
    DeepSeek 后端函数, 单线程
    '''
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            tokens = llama.tokenize(data.encode('utf-8'))

            for token in llama.generate(tokens):
                try:
                    text = llama.detokenize([token])
                    text = text.decode('utf-8')

                    if text.endswith("友:") or text.endswith("think>"):
                        print("Output: Close")
                        raise Exception("Close")

                    # print("resp:", text)
                    await websocket.send_text(text)
                    
                except Exception as e:
                    print("(IN) Exception:", e)
                    if str(e) == "Close":
                        await websocket.close()
                        break
                    else:
                        pass
                        # raise Exception("触发中断")
                
        except Exception as e:
            print("(EX) Exception:", e)
            await websocket.close()
            break
        
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
                
                if len(long_string) > 1000:
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
        
def chat(question:str=''):
    '''
    流式访问 DeepSeek 封装函数
    '''
    return asyncio.get_event_loop().run_until_complete(chat_deepseek(question))

def request_deepseek(pf:pd.DataFrame):
    '''
    通过批量生成的问题来逐个询问 DeepSeek, 后续如果 DeekSeek 后端支持并行这里可以改成并行生成
    '''
    reply = []
    for i, question in enumerate(pf['questions'].values):
        print("开始询问 DeepSeek:", question)
        
        # 获取我们的答案, 正则表达式保正确性
        response = chat(question).strip()
        # response = re.sub(r"车辆.*?在", "车辆{}在".format(plate), str(response))
        
        reply.append(response)
        print("DeepSeek 回复信息:", response, "\n")
        
    reply_pf = pd.DataFrame({
        "reply": reply
    })
    return reply_pf


if __name__ == '__main__':
    # 运行服务器
    uvicorn.run(app, host="0.0.0.0", port=82)