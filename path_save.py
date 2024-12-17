#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pathlib import Path
from manim import *
import numpy as np
import subprocess

def setup_manim_output_folder():
    """
    设置 Manim 的输出文件夹，创建到用户桌面上的 Manim 文件夹。
    """
    username = os.getlogin()
    desktop_path = Path(f"C:/Users/{username}/Desktop")
    manim_folder = desktop_path / "Manim"
    manim_folder.mkdir(exist_ok=True)
    config.media_dir = str(manim_folder)
    return manim_folder

def merge_and_clean_videos(directory, output_file):
    """
    合并指定目录下的所有视频文件，并删除中间文件。
    """
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    video_files.sort()
    if not video_files:
        print("No video files found to merge.")
        return
    
    with open('video_list.txt', 'w', encoding='utf-8') as f:
        for video in video_files:
            f.write(f"file '{os.path.join(directory, video)}'\n")
            
    ffmpeg_command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'video_list.txt', '-c', 'copy', output_file]
    subprocess.call(ffmpeg_command)
    
    for video in video_files:
        os.remove(os.path.join(directory, video))
    os.remove('video_list.txt')
    print(f"Video merged as {output_file}, all partial video files deleted.")

