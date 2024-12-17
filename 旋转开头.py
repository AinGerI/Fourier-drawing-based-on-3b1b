import os
from pathlib import Path
from manim import *
import numpy as np
import subprocess

# 获取用户桌面路径
username = os.getlogin()
desktop_path = Path(f"C:/Users/{username}/Desktop")
manim_folder = desktop_path / "Manim"
manim_folder.mkdir(exist_ok=True)
config.media_dir = str(manim_folder)

def merge_and_clean_videos(directory, output_file):
    video_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
    video_files.sort()
    if not video_files:
        print("未找到要合并的视频文件。")
        return
    
    with open('video_list.txt', 'w', encoding='utf-8') as f:
        for video in video_files:
            f.write(f"file '{os.path.join(directory, video)}'\n")
            
    ffmpeg_command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'video_list.txt', '-c', 'copy', output_file]
    subprocess.call(ffmpeg_command)
    
    for video in video_files:
        os.remove(os.path.join(directory, video))
    os.remove('video_list.txt')
    print(f"视频已合并为 {output_file}，所有部分视频文件已删除。")

class MathematicalBeauty(ThreeDScene):
    def construct(self):
        # 1. 维度演化动画
        self.dimensional_evolution()
        
        # 2. 分形动画
        self.fractal_generation()

    def dimensional_evolution(self):
        # 标题
        title = Text("Journey of Dimension", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
    
        # 0维点
        point = Dot(ORIGIN, color=GOLD)
        dim0_text = Text("0-Dimensional", font_size=24)
        dim0_text.next_to(point, DOWN)
        self.play(Create(point), Write(dim0_text))
        self.wait(1)
    
        # 1维线
        line = Line(LEFT*2, RIGHT*2, color=GOLD)
        dim1_text = Text("1-Dimensional", font_size=24)
        dim1_text.next_to(line, DOWN)
        self.play(
            ReplacementTransform(point, line),
            ReplacementTransform(dim0_text, dim1_text)
        )
        self.wait(1)
    
        # 2维平面
        plane = Rectangle(width=4, height=3, color=GOLD, fill_opacity=0.3)
        dim2_text = Text("2-Dimensional", font_size=24)
        dim2_text.next_to(plane, DOWN)
        self.play(
            ReplacementTransform(line, plane),
            ReplacementTransform(dim1_text, dim2_text)
        )
        self.wait(1)
    
        # 在转换到3D之前，先淡出2D的文字
        self.play(FadeOut(dim2_text))
        self.wait(0.5)
    
        # 3D立方体，优化视角
        self.move_camera(phi=60*DEGREES, theta=45*DEGREES)  # 设置倾斜视角
        cube = Cube(side_length=2.5, fill_opacity=0.2, color=GOLD)
        self.play(ReplacementTransform(plane, cube))
        
        # 添加立方体旋转动画
        self.play(Rotate(cube, angle=PI/2, axis=UP), run_time=2)
        self.wait(1)
        
        # 恢复相机角度并额外旋转90度，保持立方体可见
        self.move_camera(phi=0*DEGREES, theta=-90*DEGREES)
        self.wait(0.5)
        
        # 在新视角下显示3D文字
        dim3_text = Text("3-Dimensional", font_size=24)
        dim3_text.to_edge(DOWN)
        self.play(Write(dim3_text))
        self.wait(1)
        
        # 同时淡出立方体和文字
        self.play(
            FadeOut(cube),
            FadeOut(dim3_text)
        )

    def fractal_generation(self):
        # 谢尔宾斯基三角形
        title = Text("Fractal", font_size=36, color=BLUE)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
    
        def sierpinski_triangle(order, start, size):
            if order == 0:
                # 使用顶点创建三角形
                triangle = Polygon(
                    start, 
                    start + size * RIGHT, 
                    start + size * np.array([0.5, np.sqrt(3)/2, 0]), 
                    color=GOLD,
                    fill_opacity=0.3
                )
                return triangle
            
            # 计算子三角形顶点
            p1 = start
            p2 = start + size * RIGHT
            p3 = start + size * np.array([0.5, np.sqrt(3)/2, 0])
            
            # 递归创建子三角形
            triangles = VGroup(
                sierpinski_triangle(order-1, p1, size/2),
                sierpinski_triangle(order-1, p1 + size*RIGHT/2, size/2),
                sierpinski_triangle(order-1, p1 + size*np.array([0.25, np.sqrt(3)/4, 0]), size/2)
            )
            return triangles
    
        # 调整起始点使三角形居中
        start_point = LEFT*3 + 2*DOWN
        previous_fractal = None
        
        # 创建并展示分形
        for i in range(6):
            fractal = sierpinski_triangle(i, start_point, 6)
            if previous_fractal:
                self.play(
                    ReplacementTransform(previous_fractal, fractal),
                    run_time=1.5
                )
            else:
                self.play(Create(fractal), run_time=1.5)
            self.wait(0.5)
            previous_fractal = fractal
        
        # 等待后旋转180度
        self.wait(1)
        self.play(Rotate(fractal, angle=PI, about_point=ORIGIN), run_time=2)
        self.wait(1)
        self.play(FadeOut(fractal))

# 渲染和处理视频
scene = MathematicalBeauty()
scene.render()

# 动态获取类名和输出路径
class_name = scene.__class__.__name__
video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name

# 合并视频并清理
output_video = video_output_path / f"{class_name}.mp4"
merge_and_clean_videos(str(video_output_path), str(output_video))

print(f"视频已保存至: {output_video}")

