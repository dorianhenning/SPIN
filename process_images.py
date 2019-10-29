#!/vol/bitbucket/dfh17/miniconda3/envs/detection/bin/python

# Script created by Dorian Henning on 23/10/2019

import pdb

# General imports
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from time import time
import argparse

# Detection imports
from maskrcnn_benchmark.config import cfg as maskrcnn_cfg
from detection.predictor import COCODemo

# SPIN imports
from torchvision.transforms import Normalize
from models import hmr, SMPL
from utils.true_renderer import Renderer
from utils.renderer import Renderer as orthographic_renderer
import config
from utils.imutils import crop
import constants

from utils.stream import *


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Path to input directory')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

if __name__ == "__main__":
    args = parser.parse_args()

    config_file = "../maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
    maskrcnn_cfg.merge_from_file(config_file)
    
    coco_demo = COCODemo(maskrcnn_cfg, min_image_size=400, confidence_threshold=0.7)
    
    test_img = cv2.imread("examples/det_example.jpeg")
    
    
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
    renderer = Renderer(faces=smpl.faces)
    orth_renderer = orthographic_renderer(focal_length=constants.ORTH_FOCAL_LENGTH, img_res=constants.IMG_RES, faces=smpl.faces)
    
    # Dummy image and PyPlot figures for initialization
    predictions = np.zeros((640, 480, 3), dtype='uint8')
    fig = plt.figure()
    ax = fig.gca()
    imshow(ax, predictions)
    
    
    for filename in os.listdir(args.dir):
        # Read image from directory
        color_image = cv2.imread(args.dir + "/" + filename)
        #color_image = cv2.imread(args.dir + "2019-10-23-11-31-43_sot_without_obstacles/cam0/frame000170.png")
        #color_image = cv2.imread(args.dir + "2019-10-23-11-33-44_sot_with_obstacles/cam0/frame000075.png")
      #  print(filename)
      #  continue

        # Convert images to numpy arrays
      #  depth_image = np.asanyarray(depth_frame.get_data())
      #  color_image = np.asanyarray(color_frame.get_data())
    
        all_pred = coco_demo.compute_prediction(color_image)
        top_pred = coco_demo.select_top_predictions(all_pred)
        
        human_pred = select_humans(top_pred)
        if len(human_pred) == 0:
            imshow(ax, color_image)
            num_miss += 1
            continue
    
        # Preprocess input image and generate predictions
        # for now only take first image
        img_stack, norm_img_stack = process_image(color_image, human_pred, constants.IMG_NORM_MEAN, constants.IMG_NORM_STD)
        norm_img = norm_img_stack[0]
        img = img_stack[0]
    
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices
        
        orth_camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.ORTH_FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        orth_camera_translation = orth_camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()
        
        # tranformation mesh vertices to camera for rendering
        # bounding box
        bbox = human_pred[0].cpu().numpy()

        # Calculate camera parameters for rendering
        bb_x = bbox[2] - bbox[0]
        bb_y = bbox[3] - bbox[1]
        if bb_x > bb_y: 
            bb_scale = int(bb_x)
        else: 
            bb_scale = int(bb_y)
        camera_translation = torch.stack([pred_camera[:,1],
                                          pred_camera[:,2],
                                          2 * constants.FOCAL_LENGTH[0] / (pred_camera[:,0] * bb_scale)], dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()

        dx = bbox[0] + bb_x/2 - constants.CAMERA_CENTER[0] # offset from camera center in x direction
        dy = bbox[1] + bb_y/2 - constants.CAMERA_CENTER[1] # offset from camera center in y direction

        n0 = np.array([0.0, 0.0, 1.0])
        n1 = np.array([dx,-dy,constants.FOCAL_LENGTH[0]]) # y points in negative (or down) direction
        n1 = n1 / np.linalg.norm(n1)
        k = np.cross(n0, n1) # rotation axis for Rodrigues formula
        k = k / np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        theta = np.arccos(np.dot(n0,n1))

        # Rodrigues Formula
        R = np.eye(3) + np.sin(theta) * K + (1- np.cos(theta))*K*K

        # build Transformation matrix with R & T
        cam_trans_rot = np.matmul(R, camera_translation)
        T_CB = np.eye(4)
        T_CB[:3, :3] = R
        T_CB[:3, 3] = camera_translation
        T_CB[:3,3] /= np.cos(theta)
#        T = np.concatenate((R, np.expand_dims(camera_translation, axis=1)), axis=1)
#        T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)


        # Render parametric shape
        #orth_predictions = orth_renderer(pred_vertices.copy(), orth_camera_translation.copy(), img)
        predictions = renderer(pred_vertices.copy(), T_CB.copy(), color_image)
        #pdb.set_trace()
        imshow(ax, predictions)
