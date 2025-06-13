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
import tqdm

with open("config/model_conf.yaml") as f:
    model_conf = yaml.full_load(f)


def detect(img_dir):
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

    faceDetModelHandler = FaceDetModelHandler(model, "cuda", cfg)

    for i in tqdm.trange(30000):
        image_path = os.path.join(img_dir, f"{i}.jpg")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        H, W, C = image.shape

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
            logger.error("Face detection failed!")
            logger.error(e)
            continue

        # gen result
        save_path_img = os.path.join(img_dir, f"{i}_detect.jpg")
        save_path_txt = os.path.join(img_dir, f"{i}_detect.txt")

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
                # draw one box
                box = list(map(int, bboxs[min_idx]))
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.imwrite(save_path_img, image)


def gen_landmark(img_dir):
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

    for i in tqdm.trange(30000):
        image_path = os.path.join(img_dir, f"{i}.jpg")
        image_det_txt_path = os.path.join(img_dir, f"{i}_detect.txt")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if not os.path.exists(image_det_txt_path):
            continue
        with open(image_det_txt_path, "r") as f:
            lines = f.readlines()
        try:
            line = lines[0].strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            save_path_img = os.path.join(img_dir, f"{i}_landmark.jpg")
            save_path_txt = os.path.join(img_dir, f"{i}_landmark.txt")
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
            if lower.endswith("detect.jpg") or lower.endswith("landmark.jpg"):
                continue
            if lower.endswith("jpg"):
                rel_path = os.path.relpath(os.path.join(root, file), img_dir)
                image_list.append(rel_path)
    logger.info(f"Found {len(image_list)} face images")
    # logger.info(f"Image list: {image_list}")

    # detect face
    detect(img_dir)

    # generate landmarks
    gen_landmark(img_dir)
