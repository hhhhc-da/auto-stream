from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import os
import uvicorn
import torch
from torch import nn
import json

app = FastAPI()

origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class BertClassifierModule():
    def __init__(self, num_labels=2, dropout_prop=0.3, device='cuda:0' if torch.cuda.is_available() else 'cpu', pretraind_path=os.path.abspath(os.path.join('bert-chinese'))):
        # 获取 Bert 的 Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretraind_path)
        # 获取 Bert 的模型结构
        self.config = BertConfig.from_pretrained(pretraind_path, num_labels=num_labels, hidden_dropout_prob=dropout_prop)
        self.model = BertForSequenceClassification.from_pretrained(pretraind_path, config=self.config).to(device)
        
    def get_model(self):
        return self.model
    
    def get_config(self):
        return self.config
    
    def get_tokenizer(self):
        return self.tokenizer

def transform(text, tokenizer, max_len=32):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        pad_to_max_length = True,
        return_token_type_ids=True,
        truncation=True,
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    return {
        'input_text': text,
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
    }

# BERT 分类模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"模型将运行在 {device} 上\n")

# 创建模型
bert_model = BertClassifierModule(num_labels=2, device=device, pretraind_path='E:\\pandownload1\\ML\\Workflow\\modules\\bert-chinese')
model = bert_model.get_model()
model.load_state_dict(torch.load(os.path.join('models', 'bert_model.pth'), map_location=device))
tokenizer = bert_model.get_tokenizer()
last_layer = nn.Softmax(dim=1)

label_dict = {
    "daily": 0,
    "wink": 1,
}
# 反转字典
type_dict = {}
for k, v in label_dict.items():
    type_dict[v] = k

@app.post("/predict")
async def model_predict(request: Request):
    global model, tokenizer, last_layer, device, type_dict
    
    data = await request.json()
    text = json.loads(data)["text"]
    
    with torch.no_grad():
        # 模型预测
        model.eval()
    
        # 转换文本
        inputs = transform(text, tokenizer)

        # 将输入数据移动到设备上
        input_ids = inputs['input_ids'].unsqueeze(0).to(device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
        token_type_ids = inputs['token_type_ids'].unsqueeze(0).to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = last_layer(outputs.logits)
        
        pred = torch.argmax(out, dim=1).cpu().numpy()

    return {"text": text, "type": type_dict[pred[0]], "probability": out.cpu().numpy().tolist()[0]}

if __name__ == "__main__":
    # 启动 FastAPI 服务
    uvicorn.run('emoji:app', host='0.0.0.0', port=81)