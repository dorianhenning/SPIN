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
    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
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

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
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
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img

    def render_full_img(self, vertices, camera_translation, image, bbox):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
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
        color = color.astype(np.float32) / 255.0

        crop_width = int(bbox[2] - bbox[0])
        crop_height = int(bbox[3] - bbox[1])
        if crop_width > crop_height:
            # img res is saved: 224
            new_width = 224
            new_height = int(crop_height / crop_width * 224)
            height_offset = int((224 - new_height) / 2)
            res_color = cv2.resize(color[height_offset:(height_offset + new_height), :], dsize=(crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            res_depth = cv2.resize(rend_depth[height_offset:(height_offset + new_height), :], dsize=(crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
        else:
            new_width = int(crop_width / crop_height * 224)
            new_height = 224
            width_offset = int((224 - new_width) / 2)
            res_color = cv2.resize(color[:, width_offset:(width_offset + new_width)], dsize=(crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
            res_depth = cv2.resize(rend_depth[:, width_offset:(width_offset + new_width)], dsize=(crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
        import matplotlib.pyplot as plt
        import pdb
        
        valid_mask = (res_depth > 0)[:,:,None]
        full_mask = np.zeros((image.shape[0], image.shape[1], 1))
        full_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = valid_mask

        full_color = np.ones((image.shape[0], image.shape[1], 4), dtype=np.float32)
        full_color[bbox[1]:bbox[3], bbox[0]:bbox[2]] = res_color
        full_color = (full_color * 255.0).astype(np.uint8)
        output_img = ((full_color[:, :, :3] * full_mask +
                  (1 - full_mask) * image)).astype(np.uint8)
        return output_img
