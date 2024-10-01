import cv2
import numpy as np


class DeFish:

    def __init__(self, params):
        self.defisheye_params = params
        self.mapx = self.mapy = None

    def defisheye(self, image):
        if self.mapx is None or self.mapy is None:
            # Pre-compute parameters
            old_w, old_h = self.defisheye_params["input_image_size"]
            new_h, new_w = image.shape[:2]

            cam_mtx = np.array(self.defisheye_params["camera_matrix"])
            cam_mtx[0, :] = cam_mtx[0, :] * (float(new_w) / float(old_w))
            cam_mtx[1, :] = cam_mtx[1, :] * (float(new_h) / float(old_h))

            print(f"Defisheye: {(old_w, old_h)} -> ({(new_w, new_h)})")
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
