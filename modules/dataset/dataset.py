#coding=utf-8
import torch
from torch.utils.data import Dataset
import pandas as pd

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
    
def batch_transform(text, tokenizer, max_len=32):
    inputs = tokenizer.batch_encode_plus(
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

# 数据集
class TextDataset(Dataset):
    def __init__(self, data, label, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data
        self.labels = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        
        out = transform(text, self.tokenizer, self.max_len)
        out['label'] = torch.tensor(label, dtype=torch.long)
        return out
    
# 部署集
class DeployDataset(Dataset):
    def __init__(self, pf_data:pd.DataFrame, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pf_data
        
        self.length = len(pf_data.values.tolist())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        '''
        部署用的 Dataset 没有 Label
        '''
        text = str(self.data.iloc[idx]['content'])
        if text == 'nan' or text == 'None' or text == 'none' or text == 'NAN' or text is None or not text:
            text = 'nan'
        out = transform(text, self.tokenizer, self.max_len)
        
        out['user_id'] = self.data.iloc[idx]['user_id']
        out['nickname'] = self.data.iloc[idx]['nickname']
        return out