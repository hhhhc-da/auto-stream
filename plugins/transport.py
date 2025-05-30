#coding=utf-8
import blcsdk
from blcsdk import client as cli
from blcsdk import models
from typing import Optional
import sys
import asyncio
import argparse
import requests
import json
import logging

'''
每次启动时，一定要修改 blcsdk/api.py 中的 BLC_PORT 和 BLC_TOKEN 环境变量

os.environ['BLC_PORT'] = '12450'
os.environ['BLC_TOKEN'] = 'e4e11da38CBff9955AFCDe797A6D63B6'

否则连接不上服务器，每次都不同，最终软件需要编译成插件放入 blivechat/data/plugins 目录下
'''

# 全局变量，用于控制插件的关闭事件
shut_down_event: Optional[asyncio.Event] = None
logger = logging.getLogger('blivechat.plugin.transport')

class MsgHandler(blcsdk.BaseHandler):
    '''
    消息处理类，作为附件加入到 blivechat SDK 中
    '''
    def __init__(self, options: Optional[argparse.Namespace] = None):
        super().__init__()
        
        self.options = options
        
        
    def _on_add_text(self, client: cli.BlcPluginClient, message: models.AddTextMsg, extra: models.ExtraData):
        """收到弹幕"""
        logger.info(f'Received text message: {message.content}')
        
        # 将弹幕发送到指定的服务器
        if self.options and self.options.url:
            
            json_data = json.dumps({
                "text": message.content
            })
            
            try:
                # 发送 POST 请求到指定的 URL
                response = requests.post(self.options.url, json=json_data)
                if response.status_code == 200:
                    logger.info('弹幕发送成功')
                else:
                    logger.info(f'弹幕发送失败: {response.status_code} - {response.text}')
            except requests.RequestException as e:
                logger.info(f'弹幕发送异常: {e}')
        else:
            logger.info('未配置服务器地址，无法发送弹幕')
        
def opt_parser():
    """
    命令行参数解析器
    """
    parser = argparse.ArgumentParser(description='基于 BliveChat SDK 开发的弹幕发送插件')
    parser.add_argument('--url', type=str, default='http://127.0.0.1:83/text', help='需要将弹幕发送到的服务器地址')
    return parser.parse_args()
        
    
async def init_sdk(options: Optional[argparse.Namespace] = None):
    """
    初始化blivechat SDK
    """
    logger.info('blivechat SDK init.')
    await blcsdk.init()
    
    if not blcsdk.is_sdk_version_compatible():
        raise RuntimeError('SDK version is not compatible')
    
    _msg_handler = MsgHandler(options=options)
    blcsdk.set_msg_handler(_msg_handler)
    
async def main(options: Optional[argparse.Namespace] = None) -> int:
    '''
    主程序入口
    '''
    logger.info('blivechat plugin start.')
    
    try:
        await init_sdk(options=options)
    except Exception as e:
        print(f'blivechat plugin init failed: {e}')
        sys.exit(-1)
    
    logger.info('blivechat plugin init success.')
    
    # 作为一个插件，阻塞进程即可
    global shut_down_event
    shut_down_event = asyncio.Event()
    await shut_down_event.wait()
    
    return 0
    
    
if __name__ == '__main__':
    opt = opt_parser()
    sys.exit(asyncio.run(main(opt)))