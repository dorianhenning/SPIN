#!/vol/bitbucket/dfh17/miniconda3/envs/detection/bin/python

# Script created by Dorian Henning on 23/10/2019

import pdb

# General imports
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import argparse
import trimesh

# Detection imports
from maskrcnn_benchmark.config import cfg as maskrcnn_cfg
from SPIN.detection.predictor import COCODemo

# SPIN imports
from SPIN.models import hmr, SMPL
from SPIN.utils.true_renderer import Renderer
from SPIN.utils.renderer import Renderer as orthographic_renderer
import SPIN.config
import SPIN.constants

from SPIN.utils.stream import imshow, process_image, select_humans

LIVE = False

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='Path to input directory')
parser.add_argument('--outfile', type=str, default=None, help='Filename of output images. If not set use input filename.')

if __name__ == "__main__":
    args = parser.parse_args()

    config_file = "/vol/bitbucket/dfh17/git/maskrcnn-benchmark/configs/caffe2/e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"
    maskrcnn_cfg.merge_from_file(config_file)

    coco_demo = COCODemo(maskrcnn_cfg, min_image_size=400, confidence_threshold=0.7)

    # Human Detection
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pretrained model
    model = hmr(SPIN.config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load('data/model_checkpoint.pt')
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(SPIN.config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(faces=smpl.faces)
    orth_renderer = orthographic_renderer(focal_length=SPIN.constants.ORTH_FOCAL_LENGTH, img_res=SPIN.constants.IMG_RES, faces=smpl.faces)

    # Dummy image and PyPlot figures for initialization
    if LIVE:
        predictions = np.zeros((640, 480, 3), dtype='uint8')
        fig = plt.figure()
        ax = fig.gca()

    img_count = 0

    for filename in os.listdir(args.dir + "cam0/"):
        # Read image from directory
        color_image = cv2.imread(args.dir + "cam0/" + filename)

        # Object detection step
        all_pred = coco_demo.compute_prediction(color_image)
        top_pred = coco_demo.select_top_predictions(all_pred)

        # Human detection step
        human_pred = select_humans(top_pred)
        if len(human_pred) == 0:
            # ignore image if no detection (true?)
            # This is why we don't have prediction no. 533!
            continue

        # Preprocess input image and generate predictions
        # for now only take first image
        img_stack, norm_img_stack = process_image(color_image, human_pred, SPIN.constants.IMG_NORM_MEAN, SPIN.constants.IMG_NORM_STD)
        norm_img = norm_img_stack[0]
        img = img_stack[0]
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = model(norm_img.to(device))
            pred_output = smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

#        orth_camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.ORTH_FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
#        orth_camera_translation = orth_camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()

#        pdb.set_trace()
        verts = pred_output.vertices.cpu().numpy()[0]
        points = verts[smpl.faces.flatten()]
        #np.save(args.dir + 'body_points/' + filename[:-3] + 'npy', points)

        # tranformation mesh vertices to camera for rendering
        # bounding box
        bbox = human_pred[0].cpu().numpy()

        ### ROTATION METHOD
        # Calculate camera parameters for rendering
        bb_x = bbox[2] - bbox[0]
        bb_y = bbox[3] - bbox[1]
        if bb_x > bb_y:
            bb_scale = int(bb_x)
        else:
            bb_scale = int(bb_y)

        # get correct predicted distance from camera
        # camera translation == t_BC
        camera_translation = torch.stack([pred_camera[:,1],
                                          pred_camera[:,2],
                                          2 * SPIN.constants.FOCAL_LENGTH[0] / (pred_camera[:,0] * bb_scale)], dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()

        # Compute theta between image focal point and BBox center
        dx = bbox[0] + bb_x/2 - SPIN.constants.CAMERA_CENTER[0]  # offset from camera center in x direction
        dy = bbox[1] + bb_y/2 - SPIN.constants.CAMERA_CENTER[1]  # offset from camera center in y direction

        n0 = np.array([0.0, 0.0, 1.0])
        n1 = np.array([dx,-dy,SPIN.constants.FOCAL_LENGTH[0]])  # y points in negative (or down) direction
        n1 = n1 / np.linalg.norm(n1)
        theta = np.arccos(np.dot(n0,n1))

        # Rodrigues Formula
        k = np.cross(n0, n1)  # rotation axis for Rodrigues formula
        k = k / np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

        R = np.eye(3) + np.sin(theta) * K + (1- np.cos(theta)) * K.dot(K)

        # build Transformation matrix with R & T
        # t_CB = R * t_BC
        #t_CB = np.matmul(R, camera_translation)
        t_CB = camera_translation / np.cos(theta)
        T_CB = np.eye(4)
        T_CB[:3, :3] = R
        T_CB[:3, 3] = t_CB
        #T_CB[:3,3] /= np.cos(theta)

        #np.save(args.dir + 'trafo/' + filename[:-3] + 'npy', T_CB)
        
        # Render parametric shape
#        orth_predictions = orth_renderer(pred_vertices.copy(), orth_camera_translation.copy(), img)
        predictions = renderer(pred_vertices.copy(), T_CB.copy(), color_image)
        pdb.set_trace()
#        imshow(ax, predictions)
#        new_file = args.dir + "_pred/" + filename
#        plt.imsave(new_file, (predictions).astype(np.uint8))
#        print("Saved file: {}".format(filename))

        # publish 3D information to
