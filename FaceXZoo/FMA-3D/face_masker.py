"""
@author: Yinglu Liu, Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
from random import randint
import warnings

warnings.filterwarnings("ignore")
import cv2
import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from utils import read_info
from utils.render import render_texture
from model.prnet import PRNet


class PRN:
    """Process of PRNet.
    based on:
    https://github.com/YadiraF/PRNet/blob/master/api.py
    """

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resolution = 256
        self.MaxPos = self.resolution * 1.1
        self.face_ind = np.loadtxt("../../data/uv_data/face_ind.txt").astype(np.int32)
        self.triangles = np.loadtxt("../../data/uv_data/triangles.txt").astype(np.int32)
        self.net = PRNet(3, 3)
        state_dict = torch.load(model_path)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if torch.cuda.is_available():
            self.net = self.net.to(self.device)

    def process(self, image, image_info):
        if np.max(image_info.shape) > 4:  # key points to get bounding box
            # print("using landmarks")
            kpt = image_info
            if kpt.shape[0] > 3:
                kpt = kpt.T
            left = np.min(kpt[0, :])
            right = np.max(kpt[0, :])
            top = np.min(kpt[1, :])
            bottom = np.max(kpt[1, :])
        else:  # bounding box
            # print("using bounding box")
            bbox = image_info
            left = bbox[0]
            top = bbox[1]
            right = bbox[2]
            bottom = bbox[3]
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.6)
        # crop image
        src_pts = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2],
                [center[0] - size / 2, center[1] + size / 2],
                [center[0] + size / 2, center[1] - size / 2],
            ]
        )
        DST_PTS = np.array([[0, 0], [0, self.resolution - 1], [self.resolution - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)
        cropped_image = warp(
            image, tform.inverse, output_shape=(self.resolution, self.resolution)
        )
        cropped_image = np.transpose(
            cropped_image[np.newaxis, :, :, :], (0, 3, 1, 2)
        ).astype(np.float32)
        cropped_image = torch.from_numpy(cropped_image)
        if torch.cuda.is_available():
            cropped_image = cropped_image.to(self.device)
        with torch.no_grad():
            cropped_pos = self.net(cropped_image)
        cropped_pos = cropped_pos.cpu().detach().numpy()
        cropped_pos = np.transpose(cropped_pos, (0, 2, 3, 1)).squeeze() * self.MaxPos
        # restore
        cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
        z = cropped_vertices[2, :].copy() / tform.params[0, 0]
        cropped_vertices[2, :] = 1
        vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices)
        vertices = np.vstack((vertices[:2, :], z))
        pos = np.reshape(vertices.T, [self.resolution, self.resolution, 3])
        return pos

    def get_vertices(self, pos):
        """Get all points on the face from position map"""
        all_vertices = np.reshape(pos, [self.resolution**2, -1])
        vertices = all_vertices[self.face_ind, :]
        return vertices

    def get_colors_from_texture(self, texture):
        """Get all colors from the texture"""
        all_colors = np.reshape(texture, [self.resolution**2, -1])
        colors = all_colors[self.face_ind, :]
        return colors


class FaceMasker:
    """Add a virtual mask in face.

    Attributes:
        uv_face_path(str): the path of uv_face.
        mask_template_folder(str): the directory where all mask template in.
        prn(object): PRN object, https://github.com/YadiraF/PRNet.
        template_name2ref_texture_src(dict): key is template name, value is the mask load by skimage.io.
        template_name2uv_mask_src(dict): key is template name, value is the uv_mask.
        is_aug(bool): whether or not to add some augmentaion operation on the mask.
    """

    def __init__(self, is_aug):
        """init for FaceMasker

        Args:
            is_aug(bool): whether or not to add some augmentaion operation on the mask.
        """
        self.uv_face_path = "../../data/uv_data/uv_face_mask.png"
        self.mask_template_folder = "../../data/masks/texture"  #
        self.prn = PRN("model/prnet.pth")
        (
            self.template_name2ref_texture_src,
            self.template_name2uv_mask_src,
        ) = self.get_ref_texture_src()
        self.is_aug = is_aug

    def get_ref_texture_src(self):
        """Return texture and texture mask"""
        template_name2ref_texture_src = {}  # store texture templates
        template_name2uv_mask_src = {}  # store texture template masks
        mask_template_list = os.listdir(self.mask_template_folder)
        uv_face = imread(self.uv_face_path, as_gray=True) / 255.0
        self.uv_face_mask = uv_face[:, :, np.newaxis]
        for mask_template in mask_template_list:
            mask_template_path = os.path.join(self.mask_template_folder, mask_template)
            ref_texture_src = imread(mask_template_path) / 255.0
            if ref_texture_src.shape[2] == 4:  # must 4 channel, how about 3 channel?
                uv_mask_src = ref_texture_src[:, :, 3]
                ref_texture_src = ref_texture_src[:, :, :3]
            else:
                print("Fatal error!", mask_template_path)
            # set the texture template mask corresponding to uv_face_mask to 0 to remove neck part
            # print(uv_mask_src.dtype)
            # imsave("./temp/uv_mask_src.jpg", uv_mask_src)
            uv_mask_src *= uv_face
            template_name2ref_texture_src[mask_template] = ref_texture_src
            template_name2uv_mask_src[mask_template] = uv_mask_src
        return template_name2ref_texture_src, template_name2uv_mask_src

    def add_mask(
        self, face_root, image_name2lms, image_name2template_name, masked_face_root
    ):
        for image_name, face_lms in image_name2lms.items():
            image_path = os.path.join(face_root, image_name)
            masked_face_path = os.path.join(masked_face_root, image_name)  # output path
            template_name = image_name2template_name[image_name]
            self.add_mask_one(image_path, face_lms, template_name, masked_face_path)

    # you can speed it up by a c++ version.
    def render(self, vertices, new_colors, h, w):
        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = render_texture(
            vertices.T, vis_colors.T, self.prn.triangles.T, h, w, c=1
        ).astype(np.uint8)
        face_mask = np.squeeze(face_mask > 0)
        new_image = render_texture(
            vertices.T, new_colors.T, self.prn.triangles.T, h, w, c=3
        )
        return face_mask, new_image

    def add_mask_one(self, image_path, face_lms, template_name, masked_face_path):
        """Add mask to one image.

        Args:
            image_path(str): the image to add mask.
            face_lms(str): face landmarks, [x1, y1, x2, y2, ..., x106, y106]
            template_name(str): the mask template to be added on the current image,
                                got to '/Data/mask-data' for all template.
            masked_face_path(str): the path to save masked image.
        """
        image = imread(image_path)
        ref_texture_src = self.template_name2ref_texture_src[template_name]
        uv_mask_src = self.template_name2uv_mask_src[template_name]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        [h, w, c] = image.shape
        if c == 4:
            image = image[:, :, :3]
        # 3d reconstruction -> get texture.
        # vertices is all points in the original image corresponding to the position map of the face
        pos, vertices = self.get_vertices(face_lms, image)
        image = image / 255.0  #!!
        # channel 1 and channel 2 in uv position map store x and y coordinates respectively
        texture = cv2.remap(
            image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        imsave(
            "./temp/{}.jpg".format(masked_face_path.split("/")[-1].split(".")[0]),
            texture,
        )
        imsave(
            "./temp/{}_uv.jpg".format(masked_face_path.split("/")[-1].split(".")[0]),
            texture * self.uv_face_mask,
        )
        new_texture = self.get_new_texture(ref_texture_src, uv_mask_src, texture)
        new_colors = self.prn.get_colors_from_texture(new_texture)

        # render
        face_mask, new_image = self.render(
            np.ascontiguousarray(vertices),
            np.ascontiguousarray(new_colors),
            # np.ascontiguousarray(self.prn.triangles.T.astype(np.int64)),
            h,
            w,
        )
        # face_mask, new_image = self.render_cy(
        #     np.ascontiguousarray(vertices.T),
        #     np.ascontiguousarray(new_colors.T),
        #     np.ascontiguousarray(self.prn.triangles.T.astype(np.int64)),
        #     h,
        #     w,
        # )
        face_mask = np.squeeze(np.floor(face_mask) > 0)
        # imsave("./temp/face_mask.png", face_mask); exit(0)
        tmp = new_image * face_mask[:, :, np.newaxis]
        # paste face back to original image
        new_image = (
            image * (1 - face_mask[:, :, np.newaxis])
            + new_image * face_mask[:, :, np.newaxis]
        )
        new_image = np.clip(new_image, -1, 1)  # must clip to (-1, 1)!
        print("saving file: {}".format(masked_face_path))
        imsave(masked_face_path, (new_image * 255).astype(np.uint8))

    def get_vertices(self, face_lms, image):
        """Get vertices

        Args:
            face_lms: face landmarks.
            image:[0, 255]
        """
        lms_info = read_info.read_landmark_106_array(face_lms)
        pos = self.prn.process(image, lms_info)
        vertices = self.prn.get_vertices(pos)
        return pos, vertices

    def get_new_texture(self, ref_texture_src, uv_mask_src, texture):
        """Get new texture
        Mainly for data augmentation.
        """
        x_offset = 5
        y_offset = 5
        alpha = "0.5,0.8"
        beta = 0
        erode_iter = 5

        # random augmentation
        ref_texture = ref_texture_src.copy()
        uv_mask = uv_mask_src.copy()
        if self.is_aug:
            # random flip
            if np.random.rand() > 0.5:
                ref_texture = cv2.flip(ref_texture, 1, dst=None)
                uv_mask = cv2.flip(uv_mask, 1, dst=None)
            # random scale
            if np.random.rand() > 0.5:
                x_offset = np.random.randint(x_offset)
                y_offset = np.random.randint(y_offset)
                ref_texture_temp = np.zeros_like(ref_texture)
                uv_mask_temp = np.zeros_like(uv_mask)
                target_size = (256 - x_offset * 2, 256 - y_offset * 2)
                ref_texture_temp[
                    y_offset : 256 - y_offset, x_offset : 256 - x_offset, :
                ] = cv2.resize(ref_texture, target_size)
                uv_mask_temp[y_offset : 256 - y_offset, x_offset : 256 - x_offset] = (
                    cv2.resize(uv_mask, target_size)
                )
                ref_texture = ref_texture_temp
                uv_mask = uv_mask_temp
            # random erode
            if np.random.rand() > 0.8:
                t = np.random.randint(erode_iter)
                kernel = np.ones((5, 5), np.uint8)
                uv_mask = cv2.erode(uv_mask, kernel, iterations=t)
            # random contrast and brightness
            if np.random.rand() > 0.5:
                alpha_r = [float(_) for _ in alpha.split(",")]
                alpha = (alpha_r[1] - alpha_r[0]) * np.random.rand() + alpha_r[0]
                beta = beta
                img = ref_texture * 255
                blank = np.zeros(img.shape, img.dtype)
                # dst = alpha * img + beta * blank
                dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
                ref_texture = dst.clip(0, 255) / 255
        # 将口罩纹理加到人脸纹理上，形成戴口罩的人脸纹理
        new_texture = (
            texture * (1 - uv_mask[:, :, np.newaxis])  # 人脸部分
            + ref_texture[:, :, :3] * uv_mask[:, :, np.newaxis]  # 口罩部分
        )
        imsave("./temp/masked_uv_texture.png", new_texture * self.uv_face_mask)
        return new_texture

    def get_mask_texture(self, src_image_path, seg_image_path, src_face_bbox=None):
        mask_image = imread(seg_image_path)
        src_image = imread(src_image_path)
        src_image = src_image[:, :, :3]
        # print(src_image.shape)

        # use src_image to get pos and vertices with the help of PRN model
        # bbox_info = read_info.read_bbox(src_face_bbox)
        landmark = read_info.read_landmark_106_file(src_face_bbox)
        # print(bbox_info)
        pos = self.prn.process(src_image, landmark)
        # print(pos)
        # vertices = self.prn.get_vertices(pos)

        # map mask texture to UV texture map
        mask_image = mask_image / 255.0
        # https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function
        mask_UV = cv2.remap(
            mask_image,
            pos[:, :, :2].astype(np.float32),
            None,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0),
        )
        mask_UV = mask_UV.astype(np.float32)
        # mask_UV = cv2.cvtColor(mask_UV, cv2.COLOR_RGB2RGBA)
        uv_face = imread(self.uv_face_path, as_gray=True) / 255.0
        mask_UV[:, :, 3] *= uv_face
        pos_map = pos * uv_face[:, :, np.newaxis]
        return (mask_UV * 255).astype(np.uint8), (pos_map * 255).astype(np.uint8)
