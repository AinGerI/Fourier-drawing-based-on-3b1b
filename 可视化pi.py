import math
from path_save import setup_manim_output_folder, merge_and_clean_videos
from manim import *
import numpy as np

# 设置 Manim 的输出路径
manim_folder = setup_manim_output_folder()

class AnimationConfig:
    """动画配置类，集中管理所有样式和参数"""
    # 颜色配置
    GLOW_YELLOW = "#FFFF00"  # 轨迹的发光颜色
    ARROW_COLOR = "#FFFFFF"  # 箭头颜色
    CIRCLE_COLOR = "#FFFFFF"  # 旋转圆的颜色改为白色
    
    # 线条和轨迹的样式
    TRACE_STROKE_WIDTH = 1.0    # 轨迹线条的粗细
    VECTOR_STROKE_WIDTH = 2.0   # 箭头线条的粗细
    CIRCLE_STROKE_WIDTH = 2.0   # 旋转圆的线条粗细
    CIRCLE_OPACITY = 0.3        # 旋转圆的透明度
    
    # 运动参数 - 分别设置两个向量的半径
    PRIMARY_RADIUS = 2.0        # 第一个向量的半径
    SECONDARY_RADIUS = 0.4      # 第二个向量的半径
    SHOW_CIRCLES = True         # 是否显示旋转圆
    
    # 箭头参数
    ARROW_TIP_LENGTH = 0.2     # 箭头尖端长度
    ARROW_TIP_RATIO = 0.2      # 箭头尖端与线条长度的比例
    
    # 坐标轴参数
    AXIS_COLOR = GREY
    AXIS_RANGE = [-3, 3]       # 坐标轴范围
    
    # 动画参数
    ANIMATION_DURATION = 30     # 动画持续时间（秒）
    ROTATION_CYCLES = 40       # 旋转周期数
    TRACE_DENSITY = 1e-10        # 轨迹点的密度（值越小，轨迹越平滑）

class PointCalculator:
    """负责计算运动点的位置"""
    def __init__(self, primary_radius, secondary_radius):
        self.primary_radius = primary_radius
        self.secondary_radius = secondary_radius
    
    def calculate_primary_position(self, theta):
        """计算第一个旋转点的位置"""
        return np.array([
            self.primary_radius * np.cos(theta),
            self.primary_radius * np.sin(theta),
            0
        ])
    
    def calculate_secondary_position(self, theta, primary_pos):
        """计算第二个旋转点的位置"""
        return primary_pos + self.secondary_radius * np.array([
            np.cos(PI * theta),
            np.sin(PI * theta),
            0
        ])

class PIvison(MovingCameraScene):
    def create_axes(self):
        """创建坐标轴"""
        return Axes(
            x_range=[AnimationConfig.AXIS_RANGE[0], AnimationConfig.AXIS_RANGE[1]],
            y_range=[AnimationConfig.AXIS_RANGE[0], AnimationConfig.AXIS_RANGE[1]],
            axis_config={"color": AnimationConfig.AXIS_COLOR}
        )
    
    def create_arrow(self, start, end):
        """创建带箭头的连接线"""
        return Line(
            start, end,
            color=AnimationConfig.ARROW_COLOR,
            stroke_width=AnimationConfig.VECTOR_STROKE_WIDTH
        ).add_tip(
            tip_length=AnimationConfig.ARROW_TIP_LENGTH,
            tip_width=AnimationConfig.ARROW_TIP_LENGTH/2
        )
    
    def create_rotation_circles(self):
        """创建旋转圆"""
        # 第一个旋转圆（以原点为中心）
        primary_circle = Circle(
            radius=AnimationConfig.PRIMARY_RADIUS,
            color=AnimationConfig.CIRCLE_COLOR,
            stroke_width=AnimationConfig.CIRCLE_STROKE_WIDTH,
            stroke_opacity=AnimationConfig.CIRCLE_OPACITY
        )
        
        # 创建第二个旋转圆
        secondary_circle = Circle(
            radius=AnimationConfig.SECONDARY_RADIUS,
            color=AnimationConfig.CIRCLE_COLOR,
            stroke_width=AnimationConfig.CIRCLE_STROKE_WIDTH,
            stroke_opacity=AnimationConfig.CIRCLE_OPACITY
        )
        
        def update_secondary_circle(mob, primary_pos):
            mob.move_to(primary_pos)
        
        return primary_circle, secondary_circle, update_secondary_circle
    
    def setup_trace_and_lines(self, calculator, t):
        """设置轨迹和连接线"""
        def get_primary_position():
            return calculator.calculate_primary_position(t.get_value())
        
        def get_secondary_position():
            primary_pos = get_primary_position()
            return calculator.calculate_secondary_position(t.get_value(), primary_pos)
        
        # 创建轨迹，使用更高的点密度
        path = VMobject(
            color=AnimationConfig.GLOW_YELLOW,
            stroke_width=AnimationConfig.TRACE_STROKE_WIDTH
        )
        path.set_points_smoothly([get_secondary_position()])
        
        # 设置轨迹更新器
        def update_path(mob):
            new_point = get_secondary_position()
            if len(mob.points) == 0:
                mob.start_new_path(new_point)
            else:
                mob.add_smooth_curve_to(new_point)
        
        # 创建并设置箭头
        line1 = self.create_arrow(ORIGIN, get_primary_position())
        line2 = self.create_arrow(get_primary_position(), get_secondary_position())
        
        # 创建旋转圆
        if AnimationConfig.SHOW_CIRCLES:
            primary_circle, secondary_circle, update_secondary_circle = self.create_rotation_circles()
            
            def update_lines_and_circle(group):
                line1, line2, secondary_circle = group
                primary_pos = get_primary_position()
                secondary_pos = get_secondary_position()
                line1.become(self.create_arrow(ORIGIN, primary_pos))
                line2.become(self.create_arrow(primary_pos, secondary_pos))
                update_secondary_circle(secondary_circle, primary_pos)
            
            # 设置更新器
            path.add_updater(update_path)
            lines_and_circle = VGroup(line1, line2, secondary_circle)
            lines_and_circle.add_updater(update_lines_and_circle)
            
            return path, lines_and_circle, primary_circle
        else:
            def update_lines(lines):
                line1, line2 = lines
                primary_pos = get_primary_position()
                secondary_pos = get_secondary_position()
                line1.become(self.create_arrow(ORIGIN, primary_pos))
                line2.become(self.create_arrow(primary_pos, secondary_pos))
            
            # 设置更新器
            path.add_updater(update_path)
            lines = VGroup(line1, line2)
            lines.add_updater(update_lines)
            
            return path, lines, None
    
    def construct(self):
        # 初始化计算器和时间追踪器
        calculator = PointCalculator(
            AnimationConfig.PRIMARY_RADIUS,
            AnimationConfig.SECONDARY_RADIUS
        )
        t = ValueTracker(0)
        
        # 创建场景元素
        axes = self.create_axes()
        path, lines_group, primary_circle = self.setup_trace_and_lines(calculator, t)
        
        # 添加所有元素到场景
        if AnimationConfig.SHOW_CIRCLES and primary_circle:
            self.add(axes, primary_circle, path, lines_group)
        else:
            self.add(axes, path, lines_group)
        
        # 运行动画
        self.play(
            t.animate.set_value(AnimationConfig.ROTATION_CYCLES * PI),
            rate_func=linear,
            run_time=AnimationConfig.ANIMATION_DURATION
        )

# 渲染和处理视频
if __name__ == "__main__":
    scene = PIvison()
    scene.render()
    class_name = scene.__class__.__name__
    video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name
    output_video = video_output_path / f"{class_name}.mp4"
    merge_and_clean_videos(str(video_output_path), str(output_video))
    print(f"视频保存在: {output_video}")

