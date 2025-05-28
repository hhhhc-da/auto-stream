#coding=utf-8
import ChatTTS
import torch
from pydub import AudioSegment
import os
import uuid
import pandas as pd
import numpy as np
import pickle

# 全局变量
chat = ChatTTS.Chat()
chat.load(compile=False)

audio_names = []

# 基本声音生成
def fundamental_generate_voice(texts):
    """
    基础语音生成函数
    """
    # 使用ChatTTS进行语音生成
    wavs = chat.infer(texts, lang="zh")
    return wavs

# 高级声音生成
def advanced_generate_voice(texts, temperature = .3, top_P = 0.7, top_K = 20, prompt='[oral_2][laugh_0][break_6]'):
    """
    高级语音生成函数
    """
    # 使用ChatTTS进行语音生成, 先选择 speaker
    rand_spk = None
    if not os.path.exists(os.path.join("utils", "speaker.ini")):
        rand_spk = chat.sample_random_speaker() # str 类型
        pickle.dump(rand_spk, open(os.path.join("utils", "speaker.ini"), 'wb+'))
    else:
        rand_spk = pickle.load(open(os.path.join("utils", "speaker.ini"), 'rb'))

    # 语音生成参数
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb = rand_spk, # speaker 选择
        temperature = temperature,   # 随机控制性 - 越小，生成的语音越稳定、可预测；越大则越富有变化（可能包含更多情感或口音）。
        top_P = top_P,        # 核采样概率阈值 P 值
        top_K = top_K,         # 候选词数量上限 K 值
    )

    # 可用选项: oral_(0-9), laugh_(0-2), break_(0-7) 
    # 口语化程度、笑声、停顿等参数控制, 由 0 开始程度逐渐增大
    params_refine_text = ChatTTS.Chat.RefineTextParams(
        prompt=prompt,
    )

    wavs = chat.infer(texts, params_refine_text=params_refine_text, params_infer_code=params_infer_code, lang="zh")    
    return wavs

# 保存为MP3的函数
def save_audio_to_mp3(wav_tensor, sample_rate=24000, output_path=os.path.join("audio", f"output_{uuid.uuid1()}.mp3")):
    """
    将PyTorch音频张量保存为MP3文件
    """
    # 将张量转换为numpy数组
    if isinstance(wav_tensor, torch.Tensor):
        wav_numpy = wav_tensor.cpu().numpy() if wav_tensor.is_cuda else wav_tensor.numpy()
    else:
        wav_numpy = wav_tensor
    
    # 确保音频数据在[-1, 1]范围内
    if wav_numpy.max() > 1.0 or wav_numpy.min() < -1.0:
        wav_numpy = wav_numpy / np.max(np.abs(wav_numpy))
    
    # 转换为int16格式（pydub要求）
    wav_int16 = (wav_numpy * 32767).astype(np.int16)
    
    # 创建AudioSegment对象
    audio_segment = AudioSegment(
        wav_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=wav_int16.dtype.itemsize,
        channels=1  # 单声道
    )
    
    # 保存为MP3
    audio_segment.export(output_path, format="mp3")
    audio_names.append(output_path)
    print(f"音频已保存为: {output_path}")


if __name__ == "__main__":
    # 确保音频目录存在并清空
    try:
        if os.path.exists("audio") is False:
            os.makedirs("audio")    
        if len(os.listdir("audio")) > 0:
            for file in os.listdir("audio"):
                os.remove(os.path.join("audio", file))
                
        texts = ["这是一段测试文字。"]

        # 生成测试
        # wavs = fundamental_generate_voice(texts)
        wavs = advanced_generate_voice(texts, temperature=0.3, top_P=0.7, top_K=20, prompt='[oral_2][laugh_0][break_2]')
        
        for i in range(len(wavs)):
            save_audio_to_mp3(wavs[i])
    except KeyboardInterrupt:
        print("程序被手动中断\n")
    except Exception as e:
        print(f"发生错误: {e}\n")
    finally:
        print(pd.DataFrame(audio_names, columns=["filename"]), '\n')
        # # 清理音频目录
        # if os.path.exists("audio"):
        #     for file in os.listdir("audio"):
        #         os.remove(os.path.join("audio", file))
        # else:
        #     print("音频目录不存在")
    print("程序退出")