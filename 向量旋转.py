from path_save import setup_manim_output_folder, merge_and_clean_videos
from manim import *
import numpy as np

class ComplexRotations(Scene):
    def construct(self):
        # 设置坐标系
        plane = ComplexPlane().add_coordinates()
        self.add(plane)

        # 辅助函数：创建坐标标签
        def create_coord_label(mob):
            return DecimalNumber(
                0,
                num_decimal_places=2,
                include_sign=True,
            ).add_updater(lambda m: m.next_to(mob.get_tip(), UR, buff=0.1).set_value(
                complex(mob.get_end()[0], mob.get_end()[1])
            ))

        # 辅助函数：旋转向量
        def rotate_vector(v, angle):
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
            return np.dot(rotation_matrix, v)

        # 辅助函数：创建角度指示器
        def create_angle_indicator(v1, v2):
            angle = Angle(v1, v2, radius=0.5, color=YELLOW)
            label = MathTex(r"90^\circ").next_to(angle, RIGHT)
            return VGroup(angle, label)

        # 1. 单位向量旋转
        unit_vector = Arrow(ORIGIN, RIGHT, buff=0, color=BLUE)
        unit_label = create_coord_label(unit_vector)
        self.play(Create(unit_vector), Create(unit_label))
        self.wait()

        for _ in range(4):
            new_end = rotate_vector(unit_vector.get_end(), PI/2)
            new_vector = Arrow(ORIGIN, new_end, buff=0, color=BLUE)
            
            angle_indicator = create_angle_indicator(unit_vector, new_vector)
            
            self.play(
                Transform(unit_vector, new_vector),
                Create(angle_indicator),
                run_time=1,
                rate_func=smooth
            )
            
            self.wait()
            
            self.play(
                FadeOut(angle_indicator),
                run_time=0.5
            )

        self.play(FadeOut(unit_vector), FadeOut(unit_label))

        # 2. 特定向量 (1.5+2.7i) 旋转
        start = np.array([1.5, 2.7, 0])
        arbitrary_vector = Arrow(ORIGIN, start, buff=0, color=GREEN)
        arb_label = create_coord_label(arbitrary_vector)
        self.play(Create(arbitrary_vector), Create(arb_label))
        self.wait()

        for _ in range(4):
            new_end = rotate_vector(arbitrary_vector.get_end(), PI/2)
            new_vector = Arrow(ORIGIN, new_end, buff=0, color=GREEN)
            
            angle_indicator = create_angle_indicator(arbitrary_vector, new_vector)
            
            self.play(
                Transform(arbitrary_vector, new_vector),
                Create(angle_indicator),
                run_time=1,
                rate_func=smooth
            )
            
            self.wait()
            
            self.play(
                FadeOut(angle_indicator),
                run_time=0.5
            )

        self.play(FadeOut(arbitrary_vector), FadeOut(arb_label))

# 设置 Manim 的输出路径
manim_folder = setup_manim_output_folder()

# 渲染和处理视频
if __name__ == "__main__":
    scene = ComplexRotations()
    scene.render()
    class_name = scene.__class__.__name__
    video_output_path = manim_folder / "videos" / "1080p60" / "partial_movie_files" / class_name
    output_video = video_output_path / f"{class_name}.mp4"
    merge_and_clean_videos(str(video_output_path), str(output_video))
    print(f"视频保存在: {output_video}")

