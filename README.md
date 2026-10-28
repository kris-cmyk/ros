
import cv2
import numpy as np
import pyrealsense2 as rs
import time

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color, 640, 480, rs.format.bgr8, 30
)
config.enable_stream(
    rs.stream.depth, 640, 480, rs.format.z16, 30
)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

print("D455 connected. Press Q to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=0.04),
            cv2.COLORMAP_JET
        )

        h, w = depth.shape
        distance = depth_frame.get_distance(w // 2, h // 2)

        cv2.putText(
            color,
            f"Center distance: {distance:.2f} m",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 255, 0), 2
        )

        dashboard = np.hstack((color, depth_colormap))
        cv2.imshow("D455 Live Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
