import logging
import os
from typing import Dict, List

import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS, delete_file
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
        videos_and_frames = self._find_image_paths_and_group_by_videoname(
            root_folder=self.images_folder
        )
        logger.info(
            f"Number of images to detect: {sum(len(frames) for frames in videos_and_frames.values())}"
        )
        self._detect_containers(videos_and_frames=videos_and_frames)
        self._delete_data(videos_and_frames=videos_and_frames)

    @staticmethod
    def _find_image_paths_and_group_by_videoname(root_folder):
        videos_and_frames = {}
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    video_name, _ = os.path.basename(filename).rsplit("_frame_", 1)
                    image_path = os.path.join(foldername, filename)
                    if video_name not in videos_and_frames:
                        videos_and_frames[video_name] = [image_path]
                    else:
                        videos_and_frames[video_name].append(image_path)
        return videos_and_frames

    @staticmethod
    def _delete_data(videos_and_frames: Dict[str, List[str]]):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        videos_and_frames
            List containing the paths of the images to delete.
        """
        batch_count = 0
        for video_name, images_paths in videos_and_frames.items():
            for image_path in images_paths:
                delete_file(image_path)
                batch_count += 1
        logger.info(f"Number of images deleted: {batch_count}")

    def _detect_containers(self, videos_and_frames: Dict[str, List[str]]):
        results = self._process_batches(self.model, videos_and_frames, batch_size=1)
        logger.info(results)

    def _process_batches(self, model, videos_and_frames, batch_size):
        results = []
        for video_name, images_paths in videos_and_frames.items():
            self.inference_params["source"] = images_paths
            processed_images = 0
            for i in range(0, len(images_paths), batch_size):
                batch_paths = images_paths[i : i + batch_size]
                self.inference_params["source"] = batch_paths
                try:
                    self.inference_params["name"] = video_name + "_batch_"
                    batch_results = model(**self.inference_params)
                    results.extend(batch_results)
                    torch.cuda.empty_cache()  # Clear unused memory
                    processed_images += len(batch_paths)
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
                            raise RuntimeError(
                                "Out of memory with the smallest batch size"
                            )
                    else:
                        raise e
            logger.info(f"Number of images processed: {processed_images}")
            return results
