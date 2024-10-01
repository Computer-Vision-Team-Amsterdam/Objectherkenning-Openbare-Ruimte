from typing import Dict, List

import cv2
import numpy as np
import numpy.typing as npt


class DeFish:
    """
    Class to de-fisheye images using provided distortion correction parameters:

        {
            "camera_matrix": [[f_x, 0, c_x], [0, f_y, c_y], [ 0, 0, 1]],
            "distortion_params": [[k1, k2, p1, p2, k3]],
            "input_image_size": [width, height]  # Size of the original input image
        }

    See the [OpenCV
    documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
    for details.

    Parameters
    ----------
    params: Dict
        Distortion correction parameters.
    """

    def __init__(self, params: Dict[str, List]):
        self.defisheye_params = params
        self.mapx = self.mapy = None

    def defisheye(self, image: npt.NDArray) -> npt.NDArray:
        """
        Removes fisheye effect from the image using provided distortion
        correction parameters.

        *NOTE*: the current implementation pre-computes certain mappings and
        stores these as instance variables upon processing the first image.
        Afterwards, all images are assumed to have the same size.

        Parameters
        ----------
        image: npt.NDArray
            The image to de-fisheye.

        Returns
        -------
        The de-fisheyed image.
        """
        if self.mapx is None or self.mapy is None:
            # Pre-compute parameters
            old_w, old_h = self.defisheye_params["input_image_size"]
            new_h, new_w = image.shape[:2]

            cam_mtx = np.array(self.defisheye_params["camera_matrix"])
            cam_mtx[0, :] = cam_mtx[0, :] * (float(new_w) / float(old_w))
            cam_mtx[1, :] = cam_mtx[1, :] * (float(new_h) / float(old_h))

            print(f"De-fisheye: {(old_w, old_h)} -> ({(new_w, new_h)})")
            print(f"Scaled camera matrix:\n{cam_mtx}")

            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                cam_mtx,
                np.array(self.defisheye_params["distortion_params"]),
                (new_w, new_h),
                0,
                (new_w, new_h),
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                cam_mtx,
                np.array(self.defisheye_params["distortion_params"]),
                None,
                newcameramtx,
                (new_w, new_h),
                5,
            )

        # Undistort image
        img_dst = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        return img_dst
