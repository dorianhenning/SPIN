import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=388.5, img_res=[640, 480], camera_center=[324, 238], faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                       viewport_height=img_res[1],
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = camera_center
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], images_np[i]), (2,0,1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_pose, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_pose[0, 3] *= -1.
        camera_pose[1, 3] *= 1.


        #bb_x = bbox[2] - bbox[0]
        #bb_y = bbox[3] - bbox[1]
        #if bb_x > bb_y: 
        #    bb_scale = bb_x.cpu().numpy()
        #else: 
        #    bb_scale = bb_y.cpu().numpy()

        # correct distance to object by z = f * z' * x_pred_size / ( f' * bb_scale)
        #camera_trans[2] = self.focal_length * camera_trans[2] * 224.0 / (5000.0 * bb_scale)
        # correct camera shift: dx_c = dx_c' * bb_scale / x_pred_size
        #camera_trans[0:2] = camera_trans[0:2] * bb_scale / 224.0

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        #camera_pose = np.eye(4)
        #camera_pose[:3, 3] = camera_trans
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        #color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img

