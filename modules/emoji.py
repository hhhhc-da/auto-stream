from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from modules.bert import BertClassifierModule, run_bert_model

app = FastAPI()

origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# BERT 分类模型

# 创建根路由
@app.get("/")
async def root():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    # 启动 FastAPI 服务
    uvicorn.run(app, host='0.0.0.0', port=81)