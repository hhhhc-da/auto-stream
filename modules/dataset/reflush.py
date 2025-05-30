import pandas as pd
import os
import numpy as np

# 缓冲
cache = {
    "label": [],
    "content": [],
}

with open(os.path.join("stream", "label.txt"), 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        if len(line) == 0:
            continue
        
        pk = line.split('\t')
        if len(pk) != 2:
            print(f"Error: {line}")
            continue
        
        cache["label"].append(pk[0])
        cache["content"].append(pk[1])
        
    f.close()
    
pf = pd.DataFrame(cache)
key = list(pf['label'].unique())
group = pf.groupby("label")

# 创建组合字典
group_dict = {}
for k in key:
    group_dict[k] = group.get_group(k)
    
# 统计每个组的大小
sizes = [_pf.size//2 for _pf in group_dict.values()]
avg = np.mean(sizes)
print("采样数据个数:", sizes, "\n最终决定使用:", avg)

sampled_dict = {}
for k in key:
    if group_dict[k].size//2 > avg + 5:
        sampled_dict[k] = group_dict[k].sample(n=int(avg), random_state=42)
    else:
        sampled_dict[k] = group_dict[k]
    
print("采样数据个数统计:", [_pf.size//2 for _pf in sampled_dict.values()])

# # 将所有采样数据拼合到一个表内
# sampled_pf = pd.concat(sampled_dict.values(), ignore_index=True)
# sampled_pf['label'] = sampled_pf['label'].apply(lambda x: "冗余信息" if x not in ['实际用户', '需求用户'] else "需求用户")

# # 重新采样, 这次不考虑生成质量
# group = sampled_pf.groupby("label")

# key = list(sampled_pf['label'].unique())
# group_dict = {}
# for k in key:
#     group_dict[k] = group.get_group(k)
    
# # 统计每个组的大小
# sizes = [_pf.size//2 for _pf in group_dict.values()]
# print("采样数据个数:", sizes, "\n最终决定使用:", min(sizes))

# sampled_dict = {}
# for k in key:
#     sampled_dict[k] = group_dict[k].sample(n=min(sizes), random_state=42)
    
# print("采样数据个数统计:", [_pf.size//2 for _pf in sampled_dict.values()])

# 将所有采样数据拼合到一个表内
sampled_pf = pd.concat(sampled_dict.values(), ignore_index=True)

# 打乱数据
sampled_pf = sampled_pf.sample(frac=1, random_state=42).reset_index(drop=True)
sampled_pf.to_csv(os.path.join("stream", "sampled_label.txt"), sep='\t', index=False, header=False)