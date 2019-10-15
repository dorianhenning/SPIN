
# General stuff
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# Script created by Dorian Henning on 15/10/2019


# General imports
import numpy as np

# RealSense imports
import pyrealsense2 as rs

# Detection imports
from maskrcnn_benchmark.config import cfg
from detection.predictor import COCODemo


def imshow(ax, img):
    ax.imshow(img[:, :, [2, 1, 0]])
    ax.axis("off")
    plt.draw()
    plt.pause(1e-3)

config_file = "../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)

coco_demo = COCODemo(cfg, min_image_size=400, confidence_threshold=0.7)

# Dummy image and PyPlot figures for initialization
predictions = np.zeros((640, 480, 3), dtype='uint8')
fig = plt.figure()
ax = fig.gca()
imshow(ax, predictions)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        predictions = coco_demo.run_on_opencv_image(color_image)
        imshow(ax, predictions)

finally:

    # Stop streaming
    pipeline.stop()
