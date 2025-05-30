#coing=utf-8
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import io
import os
import uuid
from pydub import AudioSegment

from modules.voice_trans import voice_recognition_from_file
from modules.voice_gen import advanced_generate_voice, save_audio_to_mp3
from modules.llm_server import chat

def record_microphone(duration=5, fs=44100, output_path=os.path.join('record', f'record_{uuid.uuid1()}.mp3')):
    """
    录制系统麦克风的音频并返回 WAV 字节流
    """
    print(f"开始录制 {duration} 秒...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # 等待录音完成
    print("录制完成")
    
    # 创建内存中的文件对象
    with io.BytesIO() as f:
        # 将录音数据写入内存文件
        write(f, fs, np.int16(recording * 32767))
        # 获取字节流
        wav_bytes = f.getvalue()
        
    # 从字节流加载音频
    audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    
    # 导出为MP3
    audio.export(output_path, format="mp3")
    print(f"已保存MP3文件: {output_path}")
    return output_path

if __name__ == "__main__":
    # 设置录制参数
    duration = 5  # 录制10秒
    
    # 执行录制
    audio_path = record_microphone(duration)
    
    # 语音转文字
    try:
        text = voice_recognition_from_file(audio_path)
        print(f"识别结果: {text}")
    except Exception as e:
        print(f"语音识别失败: {e}")
    finally:
        # 删除录音文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"已删除录音文件: {audio_path}")
        
    # 使用 DeepSeek 自动回复
    try:
        response = chat(question=text)
        print(f"DeepSeek 回复: {response}")
    except Exception as e:
        print(f"DeepSeek 自动回复失败: {e}")
        
    # 将生成的文字转换为语音    
    try:
        if os.path.exists("audio") is False:
            os.makedirs("audio")    
        if len(os.listdir("audio")) > 0:
            for file in os.listdir("audio"):
                os.remove(os.path.join("audio", file))
        
        wavs = advanced_generate_voice([response], temperature=0.3, top_P=0.7, top_K=20, prompt='[oral_2][laugh_0][break_2]')
        save_audio_to_mp3(wavs[0])
        print("语音生成成功, 已保存为 MP3 文件")
    except Exception as e:
        print(f"语音生成失败: {e}")