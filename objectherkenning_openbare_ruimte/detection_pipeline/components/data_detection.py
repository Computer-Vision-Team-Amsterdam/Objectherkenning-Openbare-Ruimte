import logging
import os
from typing import Dict, List

import torch
from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths
from ultralytics import YOLO

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
        images_folder: str,
        detections_folder: str,
        model_name: str,
        pretrained_model_path: str,
        inference_params: Dict,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.images_folder = images_folder
        self.detections_folder = detections_folder
        self.model_name = model_name
        self.pretrained_model_path = os.path.join(pretrained_model_path, model_name)
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
            "project": self.detections_folder,
        }
        logger.info(f"Inference_params: {self.inference_params}")
        logger.info(f"Pretrained_model_path: {self.pretrained_model_path}")
        logger.info(f"Yolo model: {self.model_name}")
        logger.info(f"Project_path: {self.detections_folder}")

        self.model = YOLO(model=self.pretrained_model_path, task="detect")

    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects containers;
            - deletes the raw images.
        """
        logger.info(f"Running container detection pipeline on {self.images_folder}..")
        images_paths = find_image_paths(root_folder=self.images_folder)
        logger.info(f"Images paths to detect: {images_paths}")
        self._detect_containers(images_paths=images_paths)
        self._delete_data(images_paths=images_paths)

    @staticmethod
    def _delete_data(images_paths: List[str]):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        images_paths
            List containing the paths of the images to delete.
        """
        for image_path in images_paths:
            delete_file(image_path)

    def _detect_containers(self, images_paths: List[str]):
        results = self._process_batches(self.model, images_paths, batch_size=256)
        logger.info(results)

    def _process_batches(self, model, images_paths, batch_size):
        results = []
        self.inference_params["source"] = images_paths
        for i in range(0, len(images_paths), batch_size):
            batch_paths = images_paths[i : i + batch_size]
            self.inference_params["source"] = batch_paths
            try:
                batch_results = model(**self.inference_params)
                results.extend(batch_results)
                torch.cuda.empty_cache()  # Clear unused memory
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(
                        "Out of memory with batch size of {}".format(batch_size)
                    )
                    if batch_size > 1:
                        new_batch_size = batch_size // 2
                        logger.warning(
                            "Trying smaller batch size: {}".format(new_batch_size)
                        )
                        return self._process_batches(
                            model, images_paths, new_batch_size
                        )
                    else:
                        raise RuntimeError("Out of memory with the smallest batch size")
                else:
                    raise e
        return results
