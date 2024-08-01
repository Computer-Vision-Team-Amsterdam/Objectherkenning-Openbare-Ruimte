import logging
from typing import Dict, Tuple

import cv2
import numpy as np

logger = logging.getLogger("image")


class InputImage:
    mapxy = [None, None]

    def __init__(self, image_full_path: str):
        self.image = cv2.imread(str(image_full_path))

    def resize(self, output_image_size: Tuple[int, int]):
        """Resizes the image

        Parameters
        ----------
        output_image_size : List
            Output size on format: [width, height]
        """
        self.image = cv2.resize(self.image, output_image_size)

    def defisheye(self, defisheye_params: Dict):
        """Removes fisheye effect from the images.

        Parameters
        ----------
        defisheye_params : Dict
            Parameters to use for defisheying. The structure is the following:
                camera_matrix: [[2028, 0, 1954.1], [0, 2029.6, 1055.1], [ 0, 0, 1]]
                distortion_params: [[-0.24083, 0.10647, 0.00083113, 0.0001802, -0.025874]]
                input_image_size: [3840, 2160]
        """
        if self.mapxy[0] is None and self.mapxy[1] is None:
            old_w, old_h = defisheye_params["input_image_size"]
            new_h, new_w = self.image.shape[:2]

            cam_mtx = np.array(defisheye_params["camera_matrix"])
            cam_mtx[0, :] = cam_mtx[0, :] * (float(new_w) / float(old_w))
            cam_mtx[1, :] = cam_mtx[1, :] * (float(new_h) / float(old_h))

            logger.debug(f"Defisheye: {(old_w, old_h)} -> ({(new_w, new_h)})")
            logger.debug(f"Scaled camera matrix:\n{cam_mtx}")

            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                cam_mtx,
                np.array(defisheye_params["distortion_params"]),
                (new_w, new_h),
                0,
                (new_w, new_h),
            )
            self.mapxy[0], self.mapxy[1] = cv2.initUndistortRectifyMap(
                cam_mtx,
                np.array(defisheye_params["distortion_params"]),
                None,
                newcameramtx,
                (new_w, new_h),
                5,
            )

        self.image = cv2.remap(
            self.image, self.mapxy[0], self.mapxy[1], cv2.INTER_LINEAR
        )
