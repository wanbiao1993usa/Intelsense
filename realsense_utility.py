import pyrealsense2 as rs
import cv2
import numpy as np
from typing import overload, Union, Tuple


class RealSense:
    def __init__(self, width=640, height=480, fps=30, align_to_color=True, save_path=None, display_images=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.align_to_color = align_to_color
        self.save_path = save_path
        self.display_images = display_images
        self.pipeline = None
        self.profile = None

    def init_rs_stream(self):
        self.pipeline = rs.pipeline()
        config = rs.config()  # 定义配置config
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)  # 配置depth流
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)  # 配置color流
        self.profile = self.pipeline.start(config)  # 流程开始
        return self.pipeline, self.profile

    def get_aligned_images(self):
        align_to = rs.stream.color if self.align_to_color else rs.stream.depth  # 对齐方向
        align = rs.align(align_to)
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        if not aligned_depth_frame or not color_frame:
            raise ValueError("Could not obtain depth or color frames.")

        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        camera_parameters = {
            'fx': intr.fx, 'fy': intr.fy,
            'ppx': intr.ppx, 'ppy': intr.ppy,
            'height': intr.height, 'width': intr.width,
            'depth_scale': self.profile.get_device().first_depth_sensor().get_depth_scale()
        }

        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
        depth_image_3d = np.dstack((depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        return intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d

    def show_stream_depth(self):
        while True:
            intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d = self.get_aligned_images()
            print(aligned_depth_frame)
            x = self.width // 2
            y = self.height // 2
            dis = aligned_depth_frame.get_distance(x, y)  # （x, y)点的真实深度值
            print("dis: ", dis)
            camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y],
                                                                dis)  # （x, y)点在相机坐标系下的真实值，为一个三维向量。其中camera_coordinate[2]仍为dis，camera_coordinate[0]和camera_coordinate[1]为相机坐标系下的xy真实距离。
            print(camera_coordinate)

            if self.display_images:
                cv2.imshow('RGB image', color_image)  # 显示彩色图像

            if self.save_path:
                cv2.imwrite(f"{self.save_path}/color_image.png", color_image)
                cv2.imwrite(f"{self.save_path}/depth_image.png", depth_image_3d)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                self.pipeline.stop()
                break
        cv2.destroyAllWindows()
        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧

    @overload
    def get_point_depth(self, x: int, y: int):
        ...

    @overload
    def get_point_depth(self, x: float, y: float):
        ...

    def _get_pixel_coordinates(self, x: Union[int, float], y: Union[int, float]) -> Tuple[int, int]:
        """
        Convert normalized coordinates (float) to pixel coordinates (int).

        :param x: X-coordinate, can be float (normalized) or int (pixel).
        :param y: Y-coordinate, can be float (normalized) or int (pixel).
        :return: Tuple of (x, y) as pixel coordinates.
        """
        if isinstance(x, int) and isinstance(y, int):
            if 0 <= x < self.width and 0 <= y < self.height:
                return x, y
            else:
                raise ValueError("Pixel coordinates out of bounds.")
        elif isinstance(x, float) and isinstance(y, float):
            if 0 <= x <= 1 and 0 <= y <= 1:
                return int(x * self.width), int(y * self.height)
            else:
                raise ValueError("Normalized coordinates out of bounds.")
        else:
            raise TypeError("Coordinates must be both int or both float.")

    def get_point_depth(self, x: Union[int, float], y: Union[int, float]) -> float:
        """
        Get the depth at a specific point.

        :param x: X-coordinate, can be float (normalized) or int (pixel).
        :param y: Y-coordinate, can be float (normalized) or int (pixel).
        :return: Depth value at the specified point.
        """
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d = self.get_aligned_images()
        x_pixel, y_pixel = self._get_pixel_coordinates(x, y)
        distance = aligned_depth_frame.get_distance(x_pixel, y_pixel)
        return 1000*distance

    def get_average_depth(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> float:
        """
        计算一个矩形面积内的深度均值.
        :param top_left: Top-left corner of the region (x, y).
        :param bottom_right: Bottom-right corner of the region (x, y).
        :return: Average depth value in the region.
        """
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d = self.get_aligned_images()
        x1, y1 = top_left
        x2, y2 = bottom_right

        if not (0 <= x1 < x2 <= self.width and 0 <= y1 < y2 <= self.height):
            raise ValueError("Coordinates out of bounds or invalid region specified.")

        region_depth_values = depth_image[y1:y2, x1:x2]
        non_zero_depth_values = region_depth_values[region_depth_values > 0]  # 去除无效的深度值（0）

        if len(non_zero_depth_values) == 0:
            return 0.0  # 如果区域内没有有效深度值，返回0

        average_depth = np.mean(non_zero_depth_values)
        return average_depth



if __name__ == "__main__":
    vision = RealSense()
    pipeline, profile = vision.init_rs_stream()
    intr, depth_intrin, color_image, depth_image, aligned_depth_frame, depth_image_3d = vision.get_aligned_images()
    # vision.show_stream_depth()

    print(vision.get_point_depth(333, 335))
    average_depth = vision.get_average_depth((100, 200), (200, 300))
    print(average_depth)