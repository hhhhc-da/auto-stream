import requests
import os
import random

# 语音识别函数
def voice_recognition(file_path=os.path.join("audio", random.choice(os.listdir('audio'))), url="http://127.0.0.1:9977/api", language="zh", model="tiny", response_format="json"):
    '''
    文字转语音函数
    '''
    files = {
        "file": open(file_path, "rb")
    }
    
    data = {
        "language": language,
        "model": model,
        "response_format": response_format
    }
    response = requests.request("POST", url, timeout=600, data=data, files=files)
    files['file'].close()
    
    # 检查响应状态码和内容
    if response.status_code == 200:
        result = response.json()
        if result['code'] == 0:
            return str(','.join([i['text'] for i in result['data']])).strip() + '。'
        else:
            raise Exception(f"Error: {result['msg']}")
    else:
        raise Exception(f"HTTP Error: {response.status_code} - {response.text}")
    

if __name__ == "__main__":
    url = "http://127.0.0.1:9977/api"
    files = {
        "file": open(os.path.join("audio", random.choice(os.listdir('audio'))), "rb")
    }
    data = {
        "language" : "zh",
        "model" : "tiny",
        "response_format":"json"
    }
    
    response = requests.request("POST", url, timeout=600, data=data,files=files)
    files['file'].close()
    
    print(''.join([i['text'] for i in response.json()['data']]))