from .prnet import PRNet
from .arcface import arcface
from .cosface import cosface
from .facenet import facenet
from .mobileface import mobileface
from utils.image import (
    transform_for_arcface,
    transform_for_cosface,
    transform_for_facenet,
)

FACE_MODELS = {
    "arcface": {
        "name": "arcface",
        "loader": arcface,
        "transform": transform_for_arcface,
        "threshold": 0.31201186156272888,  # [-1,1]
    },
    "cosface": {
        "name": "cosface",
        "loader": cosface,
        "transform": transform_for_cosface,
        "threshold": 0.31650645124912262,  # [-1,1]
    },
    "facenet": {
        "name": "facenet",
        "loader": facenet,
        "transform": transform_for_facenet,
        "threshold": 0.4105780620574951,  # [-1,1]
    },
    "mobileface": {
        "name": "mobileface",
        "loader": mobileface,
        "transform": transform_for_cosface,  # same as cosface
        "threshold": 0.4454838879108429,  # [-1,1]
    },
}

__all__ = ["FACE_MODELS", "PRNet"]
