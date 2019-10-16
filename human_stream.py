# Script created by Dorian Henning on 15/10/2019

import pdb

# General imports
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# RealSense imports
import pyrealsense2 as rs

# Detection imports
from maskrcnn_benchmark.config import cfg as maskrcnn_cfg
from detection.predictor import COCODemo

# SPIN imports
from torchvision.transforms import Normalize
from models import hmr, SMPL
from utils.renderer import Renderer
import config
from utils.imutils import crop
import constants

from utils.stream import *


config_file = "../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
maskrcnn_cfg.merge_from_file(config_file)

coco_demo = COCODemo(maskrcnn_cfg, min_image_size=400, confidence_threshold=0.7)

test_img = cv2.imread("examples/det_example.jpeg")
all_pred = coco_demo.compute_prediction(test_img)
top_pred = coco_demo.select_top_predictions(all_pred)
print(top_pred)


human_pred = select_humans(top_pred)


# Human Detection
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pretrained model
model = hmr(config.SMPL_MEAN_PARAMS).to(device)
checkpoint = torch.load('data/model_checkpoint.pt')
model.load_state_dict(checkpoint['model'], strict=False)

# Load SMPL model
smpl = SMPL(config.SMPL_MODEL_DIR,
	batch_size=1,
	create_transl=False).to(device)
model.eval()

# Setup renderer for visualization
renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)


# Preprocess input image and generate predictions

# for now only take first image
img_stack, norm_img_stack = process_image(test_img, human_pred, constants.IMG_NORM_MEAN, constants.IMG_NORM_STD)
norm_img = norm_img_stack[0]
img = img_stack[0]

with torch.no_grad():
    pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
    pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
    pred_vertices = pred_output.vertices

# Calculate camera parameters for rendering
camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
camera_translation = camera_translation[0].cpu().numpy()
pred_vertices = pred_vertices[0].cpu().numpy()
img = img.permute(1,2,0).cpu().numpy()

# Render parametric shape
#img_shape = renderer(pred_vertices, camera_translation, img)
img_shape = renderer.render_full_img(pred_vertices, camera_translation, test_img[:,:,::-1].copy(), human_pred[0])


pdb.set_trace()
plt.imshow(cropped_detections.numpy())
exit()

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
