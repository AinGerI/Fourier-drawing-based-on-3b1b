import math  # 导入数学库，提供基本的数学函数和常量
from path_save import setup_manim_output_folder, merge_and_clean_videos  # 从自定义模块导入设置输出文件夹和合并视频的函数
from manim import *  # 导入Manim库的所有内容，用于创建动画和视频
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import sympy as sp  # 导入SymPy库，用于符号计算
from scipy.fftpack import fft  # 从SciPy库中导入快速傅里叶变换函数
import getpass  # 导入getpass库，用于获取用户名
from svgpathtools import svg2paths  # 从svgpathtools库中导入svg2paths函数，用于解析SVG文件
import os  # 导入os库，用于文件和目录操作

"""
这是一个基于Manim库的复杂动画生成程序,主要用于创建傅里叶级数绘图动画。
它可以将LaTeX公式、参数方程或SVG文件转换为动态绘制的图形。

主要功能:
1. 支持三种输入方式(第三种目前最好):
   - LaTeX公式(如数学符号)
   - 参数方程(可以绘制各种曲线)
   - SVG文件(可以绘制任意矢量图形)

2. 使用傅里叶级数分解将输入转换为旋转向量的组合
   - 自动计算傅里叶系数
   - 生成对应的旋转向量和圆圈
   - 通过向量叠加重建原始图形

3. 动画效果:
   - 显示旋转向量和跟踪圆
   - 平滑绘制路径
   - 支持相机动态跟踪
   - 多路径绘制和组合

使用方法:
1. 运行程序后根据提示选择绘制类型(1/2/3)
2. 根据选择输入相应内容:
   - LaTeX公式直接输入符号
   - 参数方程输入x(t)和y(t)表达式
   - SVG文件选择桌面上的文件

注意事项:
- 无论是选择123的哪一种，文件名都默认为：Mult_Path.mp4
- 确保安装了所需的依赖库
- SVG文件需要放在桌面上
- 参数方程支持常见数学函数
- 输出视频将保存在指定文件夹


本代码实现了一个优雅的数学可视化工具,可以帮助理解傅里叶级数和矢量图形的关系。
"""


# 设置 Manim 的输出路径
manim_folder = setup_manim_output_folder()

class Mult_Path(MovingCameraScene):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数
        # Scene parameters remain the same as before
        self.n_vectors = 100  # 设置向量的数量
        self.slow_factor = 0.25  # 设置动画的慢放因子
        self.path_stroke_width = 0.05  # 设置路径的线宽
        self.vector_stroke_width = 0.02  # 设置向量的线宽
        self.svg_scale_factor = 0.4  # SVG文件的缩放因子
        self.circle_stroke_width = 0.001  # 设置圆的线宽
        self.step_size = 0.000001  # 设置步长
        self.n_samples = 50000  # 设置采样点的数量
        
        self.symbol_color = "#FFFF00"  # 设置符号的颜色（黄色）
        self.arrow_color = "#FFFFFF"
        self.circle_color = "#E6E6E6"  # 设置圆的颜色（浅灰色）
        
        self.current_content = None  # 当前绘制的内容
        self.path_started = False  # 标记路径是否已经开始绘制
        self.is_first_point = True  # 标记是否是第一个点
        self.draw_type = None  # 绘制类型（1: LaTeX公式, 2: 参数方程, 3: SVG文件）
        self.svg_file = None  # 当前SVG文件的路径
        self.multiple_paths_mode = False  # 标记是否处于多路径模式

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
        username = getpass.getuser()  # 获取当前用户的用户名
        desktop_path = os.path.join('C:\\Users', username, 'Desktop')  # 构建桌面路径
        
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
        self.camera.frame.save_state()  # 保存相机的初始状态

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

        position = ORIGIN  # 设置初始位置为原点
        self.setup_content(position)  # 设置绘制内容
        self.create_drawing_animation()  # 创建绘制动画
        self.cleanup_and_reset_camera()  # 清理场景并重置相机
        self.wait(2)  # 等待2秒

    def process_svg_paths(self, svg_file, position):
        try:
            paths, _ = svg2paths(svg_file)  # 解析SVG文件中的路径
            if len(paths) == 0:
                raise ValueError("SVG文件中没有找到有效路径")
    
            # 计算原始的相对位置关系
            relative_positions = self.calculate_original_centers(paths)
            initial_positions = self.generate_non_overlapping_positions(len(paths))
            final_positions = self.calculate_final_positions(relative_positions)
    
            # 找到所有路径中的最大尺寸，用于统一缩放
            max_size = 0
            for path in paths:
                points = []
                for t in np.linspace(0, 1, 100):
                    point = path.point(t)
                    points.append([point.real, point.imag])
                points = np.array(points)
                size = np.abs(points).max()
                max_size = max(max_size, size)
    
            # 使用统一的缩放因子
            uniform_scale = 1.0 / max_size
    
            # 处理每个路径
            path_animations = []
            for i, (path, init_pos) in enumerate(zip(paths, initial_positions)):
                points = []
                for t in np.linspace(0, 1, self.n_samples):
                    point = path.point(t)
                    points.append([point.real, point.imag, 0])
                
                points = np.array(points)
                # 使用统一的缩放因子
                points[:, :2] -= points[:, :2].mean(axis=0)
                points[:, :2] *= uniform_scale
                points[:, :2] += init_pos[:2]
    
                # 其余代码保持不变
                coeffs = self.compute_fourier_coefficients(points)
                vectors = VGroup()
                circles = VGroup()
                current_point = points[0]
                
                for radius, freq, phase in coeffs:
                    vector = Arrow(
                        start=current_point,
                        end=current_point + RIGHT * radius,
                        color=self.arrow_color,
                        buff=0,
                        stroke_width=self.vector_stroke_width,
                        tip_length=0.08  # 确保箭头可见
                    )
                    vectors.add(vector)
                    
                    circle = Circle(
                        radius=radius,
                        stroke_width=self.circle_stroke_width,
                        stroke_color=self.circle_color
                    )
                    circle.move_to(current_point)
                    circles.add(circle)
                    
                    current_point = vector.get_end()
                
                path_animations.append((vectors, circles, coeffs, init_pos, final_positions[i]))
            
            return path_animations
            
        except Exception as e:
            print(f"处理SVG文件时出错: {e}")
            return None


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
            # 首先检查路径数量
            paths, _ = svg2paths(svg_file)
            path_count = len(paths)
            
            if path_count == 0:
                raise ValueError("SVG文件中没有找到有效路径")
            
            # 如果只有一个路径，使用原来的处理方法
            if path_count == 1:
                self.multiple_paths_mode = False
                points = []
                path = paths[0]
                for t in np.linspace(0, 1, self.n_samples):
                    point = path.point(t)
                    points.append([point.real, point.imag, 0])
                
                points = np.array(points)
                points[:, :2] -= points[:, :2].mean(axis=0)
                max_scale = np.abs(points[:, :2]).max()
                if max_scale > 0:
                    points[:, :2] /= max_scale
                points[:, :2] *= 1.5
                points[:, :2] += position[:2]
                
                return points
            else:
                # 多路径模式
                self.multiple_paths_mode = True
                return self.process_svg_paths(svg_file, position)
                
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

    def calculate_original_centers(self, paths):
        """计算并存储所有路径的原始几何中心"""
        centers = []
        for path in paths:
            points = []
            for t in np.linspace(0, 1, 100):
                point = path.point(t)
                points.append([point.real, point.imag])
            points = np.array(points)
            center = np.mean(points, axis=0)
            centers.append(center)
        
        # 计算整体中心
        overall_center = np.mean(centers, axis=0)
        # 计算相对位置（相对于整体中心的偏移）
        relative_positions = [center - overall_center for center in centers]
        
        return relative_positions

    def generate_non_overlapping_positions(self, num_paths):
        """生成不重叠的随机初始位置"""
        positions = []
        min_distance = 4  # 最小间距
        grid_size = math.ceil(math.sqrt(num_paths))  # 创建网格
        
        # 在网格中生成位置
        for i in range(num_paths):
            row = i // grid_size
            col = i % grid_size
            # 添加一些随机偏移使位置不那么规则
            x = (col - grid_size/2) * min_distance + np.random.uniform(-0.5, 0.5)
            y = (row - grid_size/2) * min_distance + np.random.uniform(-0.5, 0.5)
            positions.append(np.array([x, y, 0]))
        
        return positions
    
    def calculate_final_positions(self, relative_positions):
        """计算最终位置，保持统一的缩放比例"""
        # 使用更小的缩放因子来避免过大
        scale_factor = self.svg_scale_factor  / max(max(abs(pos[0]), abs(pos[1])) for pos in relative_positions)
        
        # 转换为场景坐标
        final_positions = []
        for rel_pos in relative_positions:
            final_pos = np.array([
                rel_pos[0] * scale_factor,
                rel_pos[1] * scale_factor,
                0
            ])
            final_positions.append(final_pos)
        
        return final_positions

    def setup_content(self, position):
        """设置绘制内容"""
        # 初始化基本属性（对所有类型都需要）
        self.time_tracker = ValueTracker(0)
        # 默认初始化 vectors 和 circles
        self.vectors = VGroup()
        self.circles = VGroup()        
    
        if self.draw_type == '3':
            points = self.get_svg_points(self.svg_file, position)
            if points is None:
                raise ValueError("无法获取SVG点")
                
            if not self.multiple_paths_mode:
                # 单路径处理
                self.path = VMobject().set_points_as_corners([position, position])
                self.path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
                
                vector_params = self.compute_fourier_coefficients(points)
                if not vector_params:
                    raise ValueError("无法计算傅里叶系数")
                
                self.vectors = VGroup()
                self.circles = VGroup()
                
                center_point = position
                for radius, freq, phase in vector_params:
                    vector = Arrow(
                        start=center_point,
                        end=center_point + RIGHT * radius,
                        color=self.arrow_color,
                        buff=0,
                        stroke_width=self.vector_stroke_width
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
            else:
                # 多路径处理
                self.path_animations = points  # 在多路径模式下，points实际上是path_animations
                
        else:
            # LaTeX和参数方程的处理
            self.path = VMobject().set_points_as_corners([position, position])
            self.path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
            
            path_points = self.get_points(position)
            if path_points is None or len(path_points) < 2:
                raise ValueError("无法获取有效的路径点")
                
            vector_params = self.compute_fourier_coefficients(path_points)
            if not vector_params:
                raise ValueError("无法计算傅里叶系数")
            
            self.vectors = VGroup()
            self.circles = VGroup()
            
            center_point = position
            for radius, freq, phase in vector_params:
                vector = Arrow(
                    start=center_point,
                    end=center_point + RIGHT * radius,
                    color=self.arrow_color,
                    buff=0,
                    stroke_width=self.vector_stroke_width
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
        # 根据绘制类型选择合适的点获取方法
        if self.draw_type == '3':
            vector_params = self.compute_fourier_coefficients(
                self.get_svg_points(self.svg_file, current_end)
            )
        else:
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

    def fit_path_points(self, points):
        """使用样条插值拟合路径点"""
        if len(points) < 2:
            return points
            
        # 使用参数化的样条插值
        t = np.linspace(0, 1, len(points))
        x = points[:, 0]
        y = points[:, 1]
        
        # 使用三次样条插值
        from scipy.interpolate import splprep, splev
        try:
            tck, _ = splprep([x, y], s=0)
            # 生成更多的点以使曲线更平滑
            t_new = np.linspace(0, 1, len(points) * 2)
            x_new, y_new = splev(t_new, tck)
            
            # 组合成新的点数组
            fitted_points = np.zeros((len(t_new), 3))
            fitted_points[:, 0] = x_new
            fitted_points[:, 1] = y_new
            return fitted_points
        except:
            # 如果插值失败，返回原始点
            return points

    def create_drawing_animation(self):
        """创建绘制动画"""
        if self.draw_type == '3' and self.multiple_paths_mode:
            # 用于存储所有绘制的路径
            drawn_paths = []
            
            for path_index, (vectors, circles, coeffs, init_pos, final_pos) in enumerate(self.path_animations):
                if not vectors or not circles:
                    print(f"警告：路径 {path_index + 1} 没有有效的向量或圆圈")
                    continue 
                
                # 重置摄像机状态
                self.camera.frame.save_state()
                
                # 创建当前路径
                current_path = VMobject()
                current_path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
                
                # 创建向量和圆圈
                self.play(
                    *[Create(v) for v in vectors],
                    *[Create(c) for c in circles],
                    run_time=2
                )
                
                # 设置摄像机跟随
                last_vector = vectors[-1]
                def update_camera(mob):
                    mob.move_to(last_vector.get_end())
                
                self.play(
                    self.camera.frame.animate.scale(0.0005).move_to(last_vector.get_end())
                )
                self.camera.frame.add_updater(update_camera)
                
                # 存储路径点
                path_points = []
                
                # 更新路径的函数
                def update_current_path(dt):
                    time = self.time_tracker.get_value() * self.slow_factor
                    current_end = vectors[0].get_start()
                    
                    for vector, circle, (radius, freq, phase) in zip(
                        vectors, circles, coeffs
                    ):
                        angle = freq * time + phase
                        end_point = current_end + np.array([
                            radius * np.cos(angle),
                            radius * np.sin(angle),
                            0
                        ])
                        vector.put_start_and_end_on(current_end, end_point)
                        circle.move_to(current_end)
                        current_end = end_point
                    
                    path_points.append(current_end)
                    
                    if len(path_points) > 10:
                        fitted_points = self.fit_path_points(np.array(path_points))
                        current_path.set_points_smoothly(fitted_points)
                
                # 添加路径和更新器
                self.add(current_path)
                self.add_updater(update_current_path)
                
                # 执行绘制动画
                self.play(
                    self.time_tracker.animate.increment_value(8.5 * PI),
                    run_time=1,
                    rate_func=linear
                )
                
                # 移除更新器
                self.remove_updater(update_current_path)
                self.camera.frame.remove_updater(update_camera)
            
                # 淡出向量和圆圈
                if path_index < len(self.path_animations) - 1:
                    self.play(
                        *[FadeOut(v) for v in vectors],
                        *[FadeOut(c) for c in circles],
                        run_time=1
                    )
                
                # 恢复相机
                self.play(Restore(self.camera.frame))
                self.camera.frame.animate.scale(0.005)
                
                # 保持路径可见并存储
                current_path.clear_updaters()
                self.add(current_path)
                drawn_paths.append(current_path)  # 存储绘制的路径
                
            # 所有路径都绘制完成后，创建组合动画
            if drawn_paths:
                # 增加所有路径的线宽
                self.play(
                    *[path.animate.set_stroke(width=1) for path in drawn_paths],
                    run_time=1
                )
            
                # 创建所有路径移动到最终位置的动画
                path_animations = []
                for path, (_, _, _, _, final_pos) in zip(drawn_paths, self.path_animations):
                    path_animations.append(
                        path.animate.move_to(final_pos)
                    )
            
                # 播放组合动画
                self.play(
                    *path_animations,
                    run_time=6,
                    rate_func=smooth
                )
                # 添加放大效果
                self.play(
                    self.camera.frame.animate.scale(0.5),  
                    run_time=6  # 动画持续 6 秒
                )
        else:
            # 单路径模式的代码保持不变
            if not self.vectors or not self.circles:
                print("警告：没有有效的向量或圆圈")
                return
            
            self.path = VMobject()
            self.path.set_stroke(width=self.path_stroke_width, color=self.symbol_color)
            self.path_started = False
            
            path_points = []
            
            def update_drawing(dt):
                time = self.time_tracker.get_value() * self.slow_factor
                current_end = self.vectors[0].get_start()
                
                for vector, circle, (radius, freq, phase) in zip(
                    self.vectors, 
                    self.circles, 
                    self.compute_fourier_coefficients(self.get_points(current_end))
                ):
                    angle = freq * time + phase
                    end_point = current_end + np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        0
                    ])
                    vector.put_start_and_end_on(current_end, end_point)
                    circle.move_to(current_end)
                    current_end = end_point
                
                path_points.append(current_end)
                
                if len(path_points) > 10:
                    fitted_points = self.fit_path_points(np.array(path_points))
                    self.path.set_points_smoothly(fitted_points)
            
            self.add(self.path)
            self.add_updater(update_drawing)


    def cleanup_and_reset_camera(self):
        """清理场景并重置相机"""
        if hasattr(self, 'multiple_paths_mode') and self.multiple_paths_mode:
            # 多路径模式清理
            all_vectors = VGroup()
            all_circles = VGroup()
            for vectors, circles, _, _, _ in self.path_animations:  # 修改这行
                all_vectors.add(vectors)
                all_circles.add(circles)
                
            self.play(
                *[FadeOut(v) for v in all_vectors],
                *[FadeOut(c) for c in all_circles],
                run_time=1
            )
        else:
            # 单路径模式清理
            if hasattr(self, 'vectors') and hasattr(self, 'circles'):
                self.play(
                    *[FadeOut(v) for v in self.vectors],
                    *[FadeOut(c) for c in self.circles],
                    run_time=1
                )
        
        self.play(Restore(self.camera.frame), run_time=1.5)

# 渲染和处理视频
if __name__ == "__main__":
    scene = Mult_Path()
    scene.render()

    class_name = scene.__class__.__name__
    video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name

    output_video = video_output_path / f"{class_name}.mp4"
    merge_and_clean_videos(str(video_output_path), str(output_video))
    print(f"视频保存在: {output_video}")


#
