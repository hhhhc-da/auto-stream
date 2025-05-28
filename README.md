# 语音对话服务

打算对接 Vtuber Studio 进行全自动化直播, 主要用于测试

### 安装教程
首先我们先安装好我们的 stt 语音转文字

```
wget https://github.com/jianchang512/stt/releases/download/v0.94/win-0.0.94.7z
```

之后我们要去下载一个大语言模型的 GGUF 文件

我使用的是 DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf，DeepSeek 1.5B 参数 Q8 量化版本

### 系统介绍

系统以网络传输为主要信息传递的方式

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