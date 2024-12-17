from path_save import setup_manim_output_folder, merge_and_clean_videos
from manim import *
import numpy as np
import sympy as sp 
from scipy.fftpack import fft
import getpass
from svgpathtools import svg2paths
import os
from svgpathtools import svg2paths

# 设置 Manim 的输出路径
manim_folder = setup_manim_output_folder()

class OnePath(MovingCameraScene):
    def __init__(self):
        super().__init__()
        # Scene parameters remain the same as before
        self.n_vectors = 100
        self.slow_factor = 0.25
        self.path_stroke_width = 0.5
        self.vector_stroke_width = 0.05
        self.circle_stroke_width = 0.01
        self.step_size = 0.00001
        self.n_samples = 50000
        
        self.symbol_color = "#FFFF00"
        self.arrow_color = "#FFFFFF"
        self.circle_color = "#EEEEEE"
        
        self.current_content = None
        self.path_started = False
        self.is_first_point = True
        self.draw_type = None
        self.svg_file = None
        self.multiple_paths_mode = False

        # Math function mapping dictionary remains the same
        self.math_functions = {
            'sin': 'np.sin',
            'cos': 'np.cos',
            'tan': 'np.tan',
            'exp': 'np.exp',
            'log': 'np.log',
            'log10': 'np.log10',
            'log2': 'np.log2',
            'sqrt': 'np.sqrt',
            'abs': 'np.abs',
            'sinh': 'np.sinh',
            'cosh': 'np.cosh',
            'tanh': 'np.tanh',
            'arcsin': 'np.arcsin',
            'arccos': 'np.arccos',
            'arctan': 'np.arctan',
            'e': 'np.e',
            'pi': 'np.pi',
        }

    def get_svg_files_from_desktop(self):
        """获取桌面上所有的SVG文件并返回文件列表"""
        username = getpass.getuser()
        desktop_path = os.path.join('C:\\Users', username, 'Desktop')
        
        # 获取所有SVG文件
        svg_files = [f for f in os.listdir(desktop_path) if f.lower().endswith('.svg')]
        
        if not svg_files:
            print("错误：在桌面上找不到任何SVG文件")
            return None
            
        print("\n在桌面上找到以下SVG文件：")
        for i, file in enumerate(svg_files, 1):
            print(f"{i}. {file}")
            
        while True:
            try:
                choice = input("\n请输入文件编号（1-" + str(len(svg_files)) + "）: ").strip()
                if not choice.isdigit() or int(choice) < 1 or int(choice) > len(svg_files):
                    print("无效的选择，请重新输入")
                    continue
                    
                selected_file = svg_files[int(choice) - 1]
                return os.path.join(desktop_path, selected_file)
                
            except (ValueError, IndexError):
                print("无效的输入，请重新选择")



    def construct(self):
        """场景主构造方法"""
        self.camera.frame.save_state()

        while True:
            draw_type = input("请选择绘制类型 (1: LaTeX公式, 2: 参数方程, 3: SVG文件): ").strip()
            if draw_type in ['1', '2', '3']:
                self.draw_type = draw_type
                break
            print("无效输入，请输入1、2或3")

        if self.draw_type == '1':
            self.current_content = input("请输入LaTeX符号 (例如 \\sum 或 \\int): ")
            try:
                MathTex(self.current_content)
            except Exception as e:
                print(f"错误：无效的LaTeX符号: {e}")
                return
        elif self.draw_type == '2':
            print("\n可用参数方程示例:")
            print("\n1. 蝴蝶曲线: ")
            print("x = sin(t) * (exp(cos(t)) - 2cos(4t) - sin(t/12)^5)")
            print("y = cos(t) * (exp(cos(t)) - 2cos(4t) - sin(t/12)^5)")
            
            print("\n2. 对数螺旋线:")
            print("x = exp(0.1t) * cos(t)")
            print("y = exp(0.1t) * sin(t)")
            
            print("\n3. 复杂曲线:")
            print("x = sin(t) * log(abs(t) + 1)")
            print("y = cos(t) * sqrt(abs(t))")
            
            print("\n可用数学函数:")
            print("基础函数: sin, cos, tan")
            print("指数和对数: exp, log(自然对数), log10, log2")
            print("双曲函数: sinh, cosh, tanh")
            print("反三角函数: arcsin, arccos, arctan")
            print("其他函数: sqrt(平方根), abs(绝对值)")
            print("常数: e(自然常数), pi(圆周率)")
            
            self.current_content = {
                'x': input("\n请输入x参数方程 (使用t作为参数): "),
                'y': input("请输入y参数方程 (使用t作为参数): ")
            }
            
            # 验证方程
            try:
                x_eq = self.replace_math_functions(self.current_content['x'])
                y_eq = self.replace_math_functions(self.current_content['y'])
                t = 0
                eval(x_eq)
                eval(y_eq)
            except Exception as e:
                print(f"错误：无效的参数方程: {e}")
                return
        else:  # SVG处理
            self.svg_file = self.get_svg_files_from_desktop()
            if not self.svg_file:
                return

        position = ORIGIN
        self.setup_content(position)
        self.create_drawing_animation()
        self.cleanup_and_reset_camera()
        self.wait(2)



    def replace_math_functions(self, equation):
        """替换方程中的数学函数为numpy版本"""
        result = equation
        # 按长度降序排序函数名，确保较长的函数名先被替换
        # 这样避免例如'sin'被替换后影响'sinh'的替换
        for func in sorted(self.math_functions.keys(), key=len, reverse=True):
            result = result.replace(func, self.math_functions[func])
        return result

    def get_points(self, position):
        """根据绘制类型生成路径点"""
        if self.draw_type == '1':
            return self.get_latex_points(self.current_content, position)
        elif self.draw_type == '2':
            return self.get_equation_points(self.current_content, position)
        else:
            return self.get_svg_points(self.svg_file, position)

    def get_latex_points(self, latex_input, position):
        """生成LaTeX符号的路径点"""
        try:
            tex = MathTex(latex_input)
            tex.height = 3
            tex.move_to(position)
        except Exception as e:
            print(f"创建LaTeX对象时出错: {e}")
            return None
        
        points = np.array(tex.get_all_points())
        if len(points) == 0:
            print("警告：无法获取符号的点")
            return None
            
        points[:, :2] -= points[:, :2].mean(axis=0)
        max_scale = np.abs(points[:, :2]).max()
        if max_scale > 0:
            points[:, :2] /= max_scale
        points[:, :2] *= 1.5
        points[:, :2] += position[:2]
        
        return self.resample_points(points)


    def get_equation_points(self, equations, position):
        """生成参数方程的路径点"""
        # 对于不同类型的方程使用不同的参数范围
        if 'log' in equations['x'] or 'log' in equations['y']:
            # 对于包含对数的方程，使用正数范围
            t = np.linspace(0.1, 6*np.pi, self.n_samples)
        else:
            t = np.linspace(0, 2*np.pi, self.n_samples)
        
        # 替换数学函数
        x_eq = self.replace_math_functions(equations['x'])
        y_eq = self.replace_math_functions(equations['y'])
        
        try:
            x = eval(x_eq)
            y = eval(y_eq)
            
            # 处理无穷大和NaN值
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            
            if len(x) == 0 or len(y) == 0:
                raise ValueError("计算结果包含无效值")
                
        except Exception as e:
            print(f"计算方程时出错: {e}")
            return None
            
        points = np.zeros((len(x), 3))
        points[:, 0] = x
        points[:, 1] = y
        
        # 归一化和缩放
        points[:, :2] -= points[:, :2].mean(axis=0)
        max_scale = np.abs(points[:, :2]).max()
        if max_scale > 0:
            points[:, :2] /= max_scale
        points[:, :2] *= 1.5
        points[:, :2] += position[:2]
        
        return points

    def get_svg_points(self, svg_file, position):
        """处理SVG文件并返回点序列"""
        try:
            paths, _ = svg2paths(svg_file)
            points = []
            
            # 收集所有路径的点
            for path in paths:
                for t in np.linspace(0, 1, self.n_samples // len(paths)):
                    point = path.point(t)
                    points.append([point.real, point.imag, 0])
            
            points = np.array(points)
            
            # 归一化和缩放处理
            points[:, :2] -= points[:, :2].mean(axis=0)
            max_scale = np.abs(points[:, :2]).max()
            if max_scale > 0:
                points[:, :2] /= max_scale
            points[:, :2] *= 1.5
            points[:, :2] += position[:2]
            
            return points
            
        except Exception as e:
            print(f"处理SVG文件时出错: {e}")
            return None

    def resample_points(self, points):
        """重采样路径点以获得均匀分布"""
        if points is None or len(points) < 2:
            return np.array([[0, 0, 0]])
            
        distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        total_length = distances[-1]
        
        if total_length == 0:
            return points
            
        even_distances = np.linspace(0, total_length, self.n_samples)
        resampled_points = np.zeros((self.n_samples, 3))
        
        for i, dist in enumerate(even_distances):
            idx = np.searchsorted(distances, dist)
            if idx == 0:
                resampled_points[i] = points[0]
            else:
                alpha = (dist - distances[idx-1]) / (distances[idx] - distances[idx-1])
                resampled_points[i] = (1-alpha) * points[idx-1] + alpha * points[idx]
                
        return resampled_points

    def compute_fourier_coefficients(self, points):
        """计算傅里叶系数"""
        if points is None or len(points) < 2:
            return []
            
        complex_points = points[:, 0] + 1j * points[:, 1]
        dt = 1.0 / len(points)
        
        coefficients = []
        for k in range(-self.n_vectors//2, self.n_vectors//2 + 1):
            coef = np.sum(
                complex_points * np.exp(-TAU * 1j * k * np.arange(len(points)) * dt)
            ) * dt
            
            if abs(coef) > 1e-5:
                coefficients.append([abs(coef), k, np.angle(coef)])
        
        return sorted(coefficients, key=lambda x: abs(x[1]))

    def setup_content(self, position):
        """设置绘制内容的向量和圆形"""
        path_points = self.get_points(position)
        if path_points is None or len(path_points) < 2:
            raise ValueError("无法获取有效的路径点")
            
        vector_params = self.compute_fourier_coefficients(path_points)
        if not vector_params:
            raise ValueError("无法计算傅里叶系数")
        
        self.vectors = VGroup()
        self.circles = VGroup()
        
        center_point = position
        for i, (radius, _, _) in enumerate(vector_params):
            vector = Arrow(
                start=center_point,
                end=center_point + RIGHT * radius,
                color=self.arrow_color,
                buff=0,
                stroke_width=self.vector_stroke_width,
                max_tip_length_to_length_ratio=0.08,
                tip_length=0.08
            )
            self.vectors.add(vector)
            
            circle = Circle(
                radius=radius,
                stroke_width=self.circle_stroke_width,
                stroke_color=self.circle_color
            )
            circle.move_to(center_point)
            self.circles.add(circle)
            
            center_point = vector.get_end()
        
        self.path = VMobject().set_points_as_corners([position, position])
        self.path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
        
        self.time_tracker = ValueTracker(0)

    def update_drawing(self, group):
        """更新绘制内容"""
        vectors, circles = group
        current_end = vectors[0].get_start()
        current_time = self.time_tracker.get_value() * self.slow_factor
        
        path_points = []
        vector_params = self.compute_fourier_coefficients(
            self.get_points(current_end)
        )
        
        if not vector_params:
            return
            
        for i, ((vector, circle), (radius, freq, phase)) in enumerate(
            zip(zip(vectors, circles), vector_params)
        ):
            angle = freq * current_time + phase
            end_point = current_end + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
            
            vector.put_start_and_end_on(current_end, end_point)
            circle.move_to(current_end)
            
            current_end = end_point
            if i == len(vectors) - 1:
                path_points.append(end_point)
                
        if path_points and len(path_points) > 0:
            if not self.path_started:
                self.path.start_new_path(path_points[-1])
                self.path.add_points_as_corners([path_points[-1]])
                self.path_started = True
            else:
                self.path.add_points_as_corners([path_points[-1]])
    
    def create_drawing_animation(self):
        """创建绘制动画"""
        self.path = VMobject()
        self.path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
        self.path_started = False
        
        # Create vectors and circles
        self.play(
            *[Create(v) for v in self.vectors],
            *[Create(c) for c in self.circles],
            run_time=2,
            rate_func=smooth
        )
        
        # Scan from first to last circle
        first_circle = self.circles[0]
        last_circle = self.circles[-1]
        
        # Move camera to first circle
        self.play(
            self.camera.frame.animate.scale(0.2).move_to(first_circle.get_center()),
            run_time=1
        )
        
        # Scan to last circle
        self.play(
            self.camera.frame.animate.move_to(last_circle.get_center()),
            run_time=10,
            rate_func=smooth
        )
        
        # Move camera back to last vector's end
        last_vector = self.vectors[-1]
        self.play(
            self.camera.frame.animate.scale(0.2).move_to(last_vector.get_end()),
            run_time=1
        )
        
        def update_camera(mob):
            mob.move_to(last_vector.get_end())
        
        self.camera.frame.add_updater(update_camera)
        
        drawing_group = VGroup(self.vectors, self.circles)
        drawing_group.add_updater(lambda g: self.update_drawing(g))
        self.add(drawing_group, self.path)
        
        self.play(
            self.time_tracker.animate.increment_value(8 * PI),
            run_time=32,
            rate_func=linear
        )
        
        self.camera.frame.remove_updater(update_camera)
        
    def cleanup_and_reset_camera(self):
        """清理场景并重置相机"""
        self.play(
            *[FadeOut(v) for v in self.vectors],
            *[FadeOut(c) for c in self.circles],
            run_time=1
        )
        
        self.play(Restore(self.camera.frame), run_time=1.5)
        

# 渲染和处理视频
if __name__ == "__main__":
    scene = OnePath()
    scene.render()

    class_name = scene.__class__.__name__
    video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name

    output_video = video_output_path / f"{class_name}.mp4"
    merge_and_clean_videos(str(video_output_path), str(output_video))
    print(f"视频保存在: {output_video}")



