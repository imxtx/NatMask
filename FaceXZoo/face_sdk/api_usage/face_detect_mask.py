"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com 
"""
import sys

sys.path.append(".")
import logging.config

logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger("api")

import math
import yaml
import cv2
import os
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from tqdm import trange

with open("config/model_conf.yaml") as f:
    model_conf = yaml.full_load(f)

if __name__ == "__main__":
    # common setting for all model, need not modify.
    model_path = "models"

    # model setting, modified along with model
    scene = "mask"
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
    
    # read image
    root = "../../data/masks/"

    for i in trange(1, 45):
        image_path = os.path.join(root, f"face/{i:02d}.png")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        H, W, C = image.shape

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
            logger.error("Face detection failed!")
            logger.error(e)
            continue

        # gen result
        save_path_img = os.path.join(root, f"bbox/{i:02d}_detect.jpg")
        save_path_txt = os.path.join(root, f"bbox/{i:02d}_detect.txt")

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
