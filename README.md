# 语音对话服务

打算对接 Vtuber Studio 进行全自动化直播, 主要用于测试

### 安装教程
首先我们先安装好我们的 stt 语音转文字

```
wget https://github.com/jianchang512/stt/releases/download/v0.94/win-0.0.94.7z
```

然后我们需要安装好 Anaconda 并创建好环境, 之后安装 requirements.txt 内的所有内容

```
pip install -r requirements.txt
```

之后我们要去下载一个大语言模型的 GGUF 文件, 并且放到 models 文件夹下, 在下载的 model 文件夹中有 stt 的 tiny 模型

我使用的是 DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf，DeepSeek 1.5B 参数 Q8 量化版本, 然后就可以使用了

### 系统介绍

系统以网络传输为主要信息传递的方式进行信息传递

首先我们启动我们的后端服务（stt 服务、DeepSeek 服务）

```
start.bat
```

之后我们启动主程序, 或者写一个前端页面用来给用户提供更友好的交互模式

```
python main.py
```

之后我们就可以看到有了下面的内容

![image](./images/valid1.png)

### 特别鸣谢

---

语音转文字项目 stt, 提供了高效的语音转文字功能

https://github.com/jianchang512/stt

---

文本转语音项目 chattts, 提供了专业的文字转语音功能

https://github.com/2noise/ChatTTS

---

大语言模型部署库 llama.cpp, 用于产生交互文本

https://github.com/ggml-org/llama.cpp