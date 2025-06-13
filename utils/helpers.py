import numpy as np
from numpy.typing import NDArray
from torch import Tensor


def read_landmark_106_array(face_lms: NDArray) -> NDArray:
    """Read the landmark 106 array"""
    # fmt: off
    map = [[1,2],[3,4],[5,6],7,9,11,[12,13],14,16,18,[19,20],21,23,25,[26,27],
           [28,29],[30,31],33,34,35,36,37,42,43,44,45,46,51,52,53,54,58,59,60,
           61,62,66,67,69,70,71,73,75,76,78,79,80,82,84,85,86,87,88,89,90,91,
           92,93,94,95,96,97,98,99,100,101,102,103]
    # fmt: on
    pts1 = np.array(face_lms, dtype=np.float64)
    pts1 = pts1.reshape((106, 2))
    pts = np.zeros((68, 2))  # map 106 to 68
    for ii in range(len(map)):
        if isinstance(map[ii], list):
            pts[ii] = np.mean(pts1[map[ii]], axis=0)
        else:
            pts[ii] = pts1[map[ii]]
    return pts


def clip_by_tensor(t: Tensor, t_min: float, t_max: float) -> Tensor:
    """
    Clip the tensor by the given minimum and maximum values
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result
