import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("inference_pipeline")


class InputImage:
    mapxy: List[Optional[cv2.typing.MatLike]] = [
        None,
        None,
    ]  # Class variable so we only need to compute it once

    def __init__(self, image_full_path: str) -> None:
        """
        This class is used to load, resize, and de-fisheye an input image.

        Parameters
        ----------
        image_full_path: str
            Path to the input image.
        """
        self.image = cv2.imread(str(image_full_path))

    def resize(self, output_image_size: Tuple[int, int]) -> None:
        """
        Resize the image if needed.

        Parameters
        ----------
        output_image_size: Tuple[int, int]
            Output size as Tuple `(width, height)`.
        """
        if (self.image.shape[0] != output_image_size[1]) or (
            self.image.shape[1] != output_image_size[0]
        ):
            self.image = cv2.resize(self.image, output_image_size)

    def defisheye(self, defisheye_params: Dict[str, List]) -> None:
        """
        Removes fisheye effect from the images using provided distortion
        correction parameters. See the [OpenCV
        documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
        for details.

        *NOTE*: the current implementation pre-computes certain mappings and
        stores these as class variables upon processing the first image.
        Afterwards, all images are assumed to have the same size and
        `defisheye_params`.

        The `defisheye_params` have the following structure:

            {
                "camera_matrix": [[f_x, 0, c_x], [0, f_y, c_y], [ 0, 0, 1]],
                "distortion_params": [[k1, k2, p1, p2, k3]], "input_image_size":
                [width, height]  # Size of the original input image
            }

        Parameters
        ----------
        defisheye_params: Dict[str, List]
            Parameters to use for distortion correction.
        """
        # Pre-compute mapx and mapy. These are stored as class variables so we
        # need to compute them only once. NOTE: this assumes from now on all
        # images will have the same size and distortion correction params.
        if (self.mapxy[0] is None) or (self.mapxy[1] is None):
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
