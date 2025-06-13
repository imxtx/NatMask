"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com
"""

import sys
import os

sys.path.append(".")
import logging.config

logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger("api")

import yaml
import cv2
import numpy as np
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import (
    FaceAlignModelHandler,
)
from tqdm import trange
from argparse import ArgumentParser

with open("config/model_conf.yaml") as f:
    model_conf = yaml.full_load(f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help="The directory of the images to be detected.",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="The directory of the bounding boxes.",
    )
    parser.add_argument(
        "--landmark_dir",
        type=str,
        required=True,
        help="The directory to save the detected images.",
    )
    args = parser.parse_args()

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

    faceAlignModelHandler = FaceAlignModelHandler(model, "cuda", cfg)

    # read image
    img_dir = args.img_dir
    bbox_dir = args.bbox_dir
    landmark_dir = args.landmark_dir

    os.makedirs(landmark_dir, exist_ok=True)

    for i in trange(30000):
        image_path = os.path.join(img_dir, f"{i}.jpg")
        image_det_txt_path = os.path.join(bbox_dir, f"{i}_detect.txt")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if not os.path.exists(image_det_txt_path):
            continue
        with open(image_det_txt_path, "r") as f:
            lines = f.readlines()
        try:
            line = lines[0].strip().split()
            det = np.asarray(list(map(int, line[0:4])), dtype=np.int32)
            landmarks = faceAlignModelHandler.inference_on_image(image, det)

            save_path_img = os.path.join(landmark_dir, f"{i}_landmark.jpg")
            save_path_txt = os.path.join(landmark_dir, f"{i}_landmark.txt")
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
