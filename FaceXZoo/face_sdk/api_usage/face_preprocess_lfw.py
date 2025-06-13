"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com
"""

import sys
import os

sys.path.append(".")
import logging

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
import logging.config

logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger("api")
import cv2
import yaml
import math
import numpy as np
from argparse import ArgumentParser

from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import (
    FaceAlignModelHandler,
)
from tqdm import tqdm

with open("config/model_conf.yaml") as f:
    model_conf = yaml.full_load(f)


def detect(img_dir, image_list):
    # common setting for all model, need not modify.
    model_path = "models"

    # model setting, modified along with model
    scene = "non-mask"
    model_category = "face_detection"
    model_name = model_conf[scene][model_category]

    logger.info("Start to load the face detection model...")
    # load model
    try:
        faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    except Exception as e:
        logger.error("Failed to parse model configuration file!")
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info("Successfully parsed the model configuration file model_meta.json!")

    try:
        model, cfg = faceDetModelLoader.load_model()
    except Exception as e:
        logger.error("Model loading failed!")
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info("Successfully loaded the face detection model!")

    faceDetModelHandler = FaceDetModelHandler(model, "cuda:7", cfg)

    for rel_path in tqdm(image_list):
        image_path = os.path.join(img_dir, rel_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        H, W, C = image.shape
        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
            logger.error("Face detection failed!")
            logger.error(e)
            continue

        # gen result
        # rel_path: Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg
        name, image_name = rel_path.strip().split("/")
        save_path_img = os.path.join(
            img_dir, f"{name}/{image_name.split('.')[0]}_detect.jpg"
        )
        save_path_txt = os.path.join(
            img_dir, f"{name}/{image_name.split('.')[0]}_detect.txt"
        )

        bboxs = dets
        with open(save_path_txt, "w") as fd:
            # get the midmost box
            pivot_x, pivot_y = W / 2, H / 2
            min_dist = 1e10
            min_idx = -1
            for i, box in enumerate(bboxs):
                center_x = (box[2] + box[0]) / 2
                center_y = (box[3] + box[1]) / 2
                dist = math.sqrt((pivot_x - center_x) ** 2 + (pivot_y - center_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i

            if len(bboxs) > 0:
                box = bboxs[min_idx]
                line = (
                    str(int(box[0]))
                    + " "
                    + str(int(box[1]))
                    + " "
                    + str(int(box[2]))
                    + " "
                    + str(int(box[3]))
                    + "\n"
                )
                fd.write(line)

        box = list(map(int, bboxs[min_idx]))
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite(save_path_img, image)


def gen_landmark(img_dir, image_list):
    # common setting for all model, need not modify.
    model_path = "models"

    # model setting, modified along with model
    scene = "non-mask"
    model_category = "face_alignment"
    model_name = model_conf[scene][model_category]

    logger.info("Start to load the face landmark model...")
    # load model
    try:
        faceAlignModelLoader = FaceAlignModelLoader(
            model_path, model_category, model_name
        )
    except Exception as e:
        logger.error("Failed to parse model configuration file!")
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info("Successfully parsed the model configuration file model_meta.json!")

    try:
        model, cfg = faceAlignModelLoader.load_model()
    except Exception as e:
        logger.error("Model loading failed!")
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info("Successfully loaded the face landmark model!")

    faceAlignModelHandler = FaceAlignModelHandler(model, "cuda:7", cfg)

    for rel_path in tqdm(image_list):
        image_path = os.path.join(img_dir, rel_path)
        # rel_path: Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg
        name, image_name = rel_path.strip().split("/")
        image_name_no_ext = image_name.split(".")[0]
        image_det_txt_path = os.path.join(
            img_dir, os.path.join(name, f"{image_name_no_ext}_detect.txt")
        )
        if not os.path.exists(image_det_txt_path):
            # print(f"File not found: {image_det_txt_path}")
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        with open(image_det_txt_path, "r") as f:
            lines = f.readlines()
            try:
                line = lines[0].strip().split()
                det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
                landmarks = faceAlignModelHandler.inference_on_image(image, det)

                save_path_img = os.path.join(
                    img_dir,
                    os.path.join(name, f"{image_name_no_ext}_landmark.jpg"),
                )
                save_path_txt = os.path.join(
                    img_dir,
                    os.path.join(name, f"{image_name_no_ext}_landmark.txt"),
                )
                image_show = image.copy()
                with open(save_path_txt, "w+") as fd:
                    for x, y in landmarks.astype(np.int32):
                        cv2.circle(image_show, (x, y), 2, (255, 0, 0), -1)
                        line = str(x) + " " + str(y) + " "
                        fd.write(line)
                cv2.imwrite(save_path_img, image_show)
            except Exception as e:
                logger.error("Face landmark failed!")
                logger.error(e)


def crop(img_dir, image_list):
    face_cropper = FaceRecImageCropper()

    for rel_path in tqdm(image_list):
        image_path = os.path.join(img_dir, rel_path)
        # rel_path: Zydrunas_Ilgauskas/Zydrunas_Ilgauskas_0001.jpg
        name, image_name = rel_path.strip().split("/")
        image_name_no_ext = image_name.split(".")[0]
        image_landmark_path = os.path.join(
            img_dir, os.path.join(name, f"{image_name_no_ext}_landmark.txt")
        )
        if not os.path.exists(image_landmark_path):
            # print(f"File not found: {image_landmark_path}")
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        with open(image_landmark_path, "r") as f:
            line = f.readline().strip()
            landmarks_str = line.split(" ")
            landmarks = [float(num) for num in landmarks_str]
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
            cropped_path = os.path.join(
                img_dir,
                os.path.join(name, f"{image_name_no_ext}_cropped.jpg"),
            )
            cv2.imwrite(cropped_path, cropped_image)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="The directory of the images to be detected.",
    )
    args = parser.parse_args()

    img_dir = args.img_dir

    image_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            lower = file.lower()
            if (
                lower.endswith("detect.jpg")
                or lower.endswith("landmark.jpg")
                or lower.endswith("cropped.jpg")
            ):
                continue
            if lower.endswith("jpg"):
                rel_path = os.path.relpath(os.path.join(root, file), img_dir)
                image_list.append(rel_path)
    logger.info(f"Found {len(image_list)} face images")
    # logger.info(f"Image list: {image_list}")

    # detect face
    detect(img_dir, image_list)

    # generate landmarks
    gen_landmark(img_dir, image_list)

    # crop
    crop(img_dir, image_list)

    # detect and generate landmarks for cropped images
    image_list = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            lower = file.lower()
            if file.lower().endswith("cropped.jpg"):
                rel_path = os.path.relpath(os.path.join(root, file), img_dir)
                image_list.append(rel_path)
    logger.info(f"Found {len(image_list)} cropped face images")
    detect(img_dir, image_list)
    gen_landmark(img_dir, image_list)

    # image_path = "api_usage/test_images/test1.jpg"
    # image_info_file = "api_usage/test_images/test1_landmark_res0.txt"
    # line = open(image_info_file).readline().strip()
    # landmarks_str = line.split(" ")
    # landmarks = [float(num) for num in landmarks_str]

    # face_cropper = FaceRecImageCropper()
    # image = cv2.imread(image_path)
    # cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    # cv2.imwrite("api_usage/temp/test1_cropped.jpg", cropped_image)
    # logger.info("Crop image successful!")
