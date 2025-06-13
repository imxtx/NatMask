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

import yaml
import cv2
import numpy as np
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler

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

    # read image

    image_folder = "api_usage/face_256/images"

    for i in range(31, 32):
        image_path = image_folder + f"/{i:03}.jpg"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        faceDetModelHandler = FaceDetModelHandler(model, "cuda:0", cfg)

        try:
            dets = faceDetModelHandler.inference_on_image(image)
        except Exception as e:
            logger.error("Face detection failed!")
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info("Face detected.")

        # gen result
        save_path_img = f"api_usage/face_256/detected/{i:03}_detect_res.jpg"
        save_path_txt = f"api_usage/face_256/detected/{i:03}_detect_res.txt"

        bboxs = dets
        with open(save_path_txt, "w") as fd:
            for box in bboxs:
                line = (
                    str(int(box[0]))
                    + " "
                    + str(int(box[1]))
                    + " "
                    + str(int(box[2]))
                    + " "
                    + str(int(box[3]))
                    # + " "
                    # + str(box[4]) # probability
                    + "\n"
                )
                fd.write(line)

        for box in bboxs:
            box = list(map(int, box))
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite(save_path_img, image)
        logger.info("Detection results saved!")
