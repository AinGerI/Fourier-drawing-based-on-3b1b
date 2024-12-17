from manim import *
import numpy as np
from path_save import setup_manim_output_folder, merge_and_clean_videos

class TrigonometricArrowAnimation(Scene):
    def construct(self):
        self.arrow_stroke_width = 2  # 箭头线条粗细
        self.arrow_tip_length = 0.2  # 箭头尖端长度
        self.arrow_max_tip_length_ratio = 0.2  # 箭头尖端长度比例
        self.show_axis()
        self.show_circle()
        self.move_arrow_and_draw_curves()
        self.wait()

    def show_axis(self):
        x_start = np.array([-6, 0, 0])
        x_end = np.array([6, 0, 0])
        y_start = np.array([-4, -2, 0])
        y_end = np.array([-4, 2, 0])
        x_axis = Line(x_start, x_end, color=WHITE)
        y_axis = Line(y_start, y_end, color=WHITE)
        self.add(x_axis, y_axis)
        self.add_x_labels()
        self.origin_point = np.array([-4, 0, 0])
        self.curve_start = np.array([-3, 0, 0])

    def add_x_labels(self):
        x_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"),
        ]
        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-1 + 2*i, 0, 0]), DOWN)
            self.add(x_labels[i])

    def show_circle(self):
        circle = Circle(radius=1.5, color=WHITE)
        circle.move_to(self.origin_point)
        self.add(circle)
        self.circle = circle

    def create_arrow(self, start, end):
        return Arrow(
            start=start, 
            end=end, 
            buff=0, 
            color=WHITE,
            stroke_width=self.arrow_stroke_width,
            max_tip_length_to_length_ratio=self.arrow_max_tip_length_ratio,
            tip_length=self.arrow_tip_length
        )

    def move_arrow_and_draw_curves(self):
        orbit = self.circle
        origin_point = self.origin_point
        arrow = self.create_arrow(origin_point, orbit.point_from_proportion(0))
        self.t_offset = 0
        rate = 0.25

        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            new_end = orbit.point_from_proportion(self.t_offset % 1)
            mob.put_start_and_end_on(origin_point, new_end)

        def get_sine_line():
            return DashedLine(arrow.get_end(), np.array([arrow.get_end()[0], self.origin_point[1], 0]), color="#8080FF")

        def get_cosine_line():
            return DashedLine(arrow.get_end(), np.array([self.origin_point[0], arrow.get_end()[1], 0]), color="#FF8080")

        def get_line_to_sine_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = arrow.get_end()[1]
            return Line(np.array([x, self.origin_point[1], 0]), np.array([x, y, 0]), color="#8080FF", stroke_width=2)

        def get_line_to_cosine_curve():
            x = self.curve_start[0] + self.t_offset * 4
            y = arrow.get_end()[0] - self.origin_point[0] + self.curve_start[1]
            return Line(np.array([x, self.origin_point[1], 0]), np.array([x, y, 0]), color="#FF8080", stroke_width=2)

        self.sine_curve = VGroup()
        self.cosine_curve = VGroup()
        self.sine_curve.add(Line(self.curve_start, self.curve_start))
        self.cosine_curve.add(Line(self.curve_start, self.curve_start))

        def get_sine_curve():
            last_line = self.sine_curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = arrow.get_end()[1]
            new_line = Line(last_line.get_end(), np.array([x, y, 0]), color="#8080FF")
            self.sine_curve.add(new_line)
            return self.sine_curve

        def get_cosine_curve():
            last_line = self.cosine_curve[-1]
            x = self.curve_start[0] + self.t_offset * 4
            y = arrow.get_end()[0] - self.origin_point[0] + self.curve_start[1]
            new_line = Line(last_line.get_end(), np.array([x, y, 0]), color="#FF8080")
            self.cosine_curve.add(new_line)
            return self.cosine_curve

        arrow.add_updater(go_around_circle)
        sine_line = always_redraw(get_sine_line)
        cosine_line = always_redraw(get_cosine_line)
        sine_curve_line = always_redraw(get_line_to_sine_curve)
        cosine_curve_line = always_redraw(get_line_to_cosine_curve)
        sine_curve = always_redraw(get_sine_curve)
        cosine_curve = always_redraw(get_cosine_curve)

        self.add(arrow, sine_line, cosine_line, sine_curve_line, cosine_curve_line, sine_curve, cosine_curve)
        self.wait(8.5)
        arrow.remove_updater(go_around_circle)

# 设置 Manim 的输出路径
manim_folder = setup_manim_output_folder()

# 渲染和处理视频
if __name__ == "__main__":
    scene = TrigonometricArrowAnimation()
    scene.render()
    class_name = scene.__class__.__name__
    video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name


