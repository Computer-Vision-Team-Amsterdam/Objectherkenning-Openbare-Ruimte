# import logging
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from objectherkenning_openbare_ruimte.inference_pipeline.source import blurring_tools

# logger = logging.getLogger("inference_pipeline")


class DataInference:
    def __init__(
        self,
        images_folder: str,
        inference_folder: str,
        model_name: str,
        pretrained_model_path: str,
        inference_params: Dict,
        target_classes: List,
        sensitive_classes: List,
        tracking_flag: bool = False,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.images_folder = images_folder
        self.inference_folder = inference_folder
        self.model_name = model_name
        self.pretrained_model_path = os.path.join(pretrained_model_path, model_name)
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
            "project": self.inference_folder,
        }
        print(f"Inference_params: {self.inference_params}")
        print(f"Pretrained_model_path: {self.pretrained_model_path}")
        print(f"Yolo model: {self.model_name}")
        print(f"Project_path: {self.inference_folder}")

        self.model = YOLO(model=self.pretrained_model_path, task="detect")
        self.roi = self.mapx = self.mapy = None
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.tracking_flag = tracking_flag

    def run_pipeline_prelabeling(self):
        """
        Runs the detection pipeline for pre-labeling or evaluation:
            - find the images to detect;
            - detects everything.
        If tracking_flag = True, it also tracks the detected objects.
        """
        print(f"Running container detection pipeline on {self.images_folder}..")
        videos_and_frames = self._find_image_paths_and_group_by_videoname(
            root_folder=self.images_folder
        )
        print(
            f"Number of images to detect: {sum(len(frames) for frames in videos_and_frames.values())}"
        )
        if self.tracking_flag:
            self.model = YOLO(model=self.pretrained_model_path, task="track")
            self._track_objects(videos_and_frames=videos_and_frames)
        else:
            self._detect_objects(videos_and_frames=videos_and_frames)

    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects and tracks containers;
        """
        print(f"Running container detection pipeline on {self.images_folder}..")
        videos_and_frames = self._find_image_paths_and_group_by_videoname(
            root_folder=self.images_folder
        )
        print(
            f"Number of images to detect: {sum(len(frames) for frames in videos_and_frames.values())}"
        )
        self._detect_containers(videos_and_frames=videos_and_frames)

    @staticmethod
    def _track_objects(self, videos_and_frames: Dict[str, List[str]]):
        _, images_paths = videos_and_frames.items()
        results = self.model.track(images_paths, **self.inference_params)
        print(results)

    @staticmethod
    def _detect_objects(self, videos_and_frames: Dict[str, List[str]]):
        batch_size = 1
        results = self._process_batches(
            self.model, videos_and_frames, batch_size=batch_size, is_prelabeling=True
        )
        print(results)

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

    def _detect_containers(self, videos_and_frames: Dict[str, List[str]]):
        batch_size = 1
        results = self._process_batches(
            self.model, videos_and_frames, batch_size=batch_size, is_prelabeling=False
        )
        print(results)

    def _process_batches(
        self, model, videos_and_frames, batch_size, is_prelabeling=False
    ):
        for video_name, images_paths in videos_and_frames.items():
            self.inference_params["source"] = images_paths
            processed_images = 0
            for i in range(0, len(images_paths), batch_size):
                try:
                    batch_images = [
                        cv2.imread(image_path)
                        for image_path in images_paths[i : i + batch_size]
                    ]
                    # batch_images = self._defisheye(batch_images)
                    self.inference_params["source"] = batch_images
                    self.inference_params["name"] = video_name + "_batch_"
                    batch_results = model(**self.inference_params)
                    print(f"Result YOLO: {batch_results}")
                    torch.cuda.empty_cache()  # Clear unused memory
                    if is_prelabeling:
                        self._process_results_objects(batch_results)
                    else:
                        self._process_results_containers(batch_results)
                    processed_images += len(batch_images)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("Out of memory with batch size of {}".format(batch_size))
                        if batch_size > 1:
                            new_batch_size = batch_size // 2
                            print(
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
            print(f"Number of images processed: {processed_images}")

    @staticmethod
    def _get_annotion_string_from_boxes(boxes: Boxes) -> str:
        boxes = boxes.cpu()

        annotation_lines = []

        for box in boxes:
            cls = int(box.cls.squeeze())
            conf = float(box.conf.squeeze())
            tracking_id = int(box.id.squeeze()) if box.is_track else -1
            yolo_box_str = " ".join([f"{x:.6f}" for x in box.xywhn.squeeze()])
            annotation_lines.append(f"{cls} {yolo_box_str} {conf:.6f} {tracking_id}")
        return "\n".join(annotation_lines)

    def _process_results_containers(self, model_results: Results):
        for r in model_results:
            result = r.cpu()
            boxes = result.boxes.numpy()

            print(f"Target classes: {self.target_classes}")
            print(f"Boxes classes: {boxes.cls}")
            print(f"Boxes: {boxes}")
            target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
            if len(target_idxs) == 0:  # Nothing to do!
                print("No container detected.")
                continue

            image = result.orig_img.copy()
            sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

            # Blur sensitive data
            sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
            image = blurring_tools.blur_inside_boxes(image, sensitive_bounding_boxes)

            # Draw annotation boxes
            target_bounding_boxes = boxes[target_idxs].xyxy
            image = blurring_tools.draw_bounding_boxes(image, target_bounding_boxes)

            # Save image
            folder_path = os.path.join(
                self.inference_folder, os.path.basename(os.path.dirname(result.path))
            )
            print(f"Folder path: {folder_path}")
            images_dir = os.path.join(self.inference_folder, "processed_images")
            os.makedirs(images_dir, exist_ok=True)
            print("Created images folder.")
            image_path = result.path  # Path of the input image
            image_name = os.path.basename(image_path)
            save_path = os.path.join(images_dir, image_name)
            print(f"Save path: {save_path}")
            cv2.imwrite(save_path, image)
            print("Saved image.")

            # Save annotation
            annotation_str = self._get_annotion_string_from_boxes(boxes[target_idxs])
            labels_dir = os.path.join(self.inference_folder, "processed_images")
            os.makedirs(labels_dir, exist_ok=True)
            print("Created labels folder.")
            annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(annotation_path, "w") as f:
                f.write(annotation_str)

    def _process_results_objects(self, model_results: Results):
        for r in model_results:
            result = r.cpu()
            boxes = result.boxes.numpy()

            print(f"Boxes classes: {boxes.cls}")
            print(f"Boxes: {boxes}")

            image = result.orig_img.copy()

            # Draw annotation boxes
            target_bounding_boxes = boxes.xyxy
            image = blurring_tools.draw_bounding_boxes(image, target_bounding_boxes)

            # Save image
            folder_path = os.path.join(
                self.inference_folder, os.path.basename(os.path.dirname(result.path))
            )
            print(f"Folder path: {folder_path}")
            images_dir = os.path.join(self.inference_folder, "processed_images")
            os.makedirs(images_dir, exist_ok=True)
            print("Created images folder.")
            image_path = result.path  # Path of the input image
            image_name = os.path.basename(image_path)
            save_path = os.path.join(images_dir, image_name)
            print(f"Save path: {save_path}")
            cv2.imwrite(save_path, image)
            print("Saved image.")

            # Save annotation
            annotation_str = self._get_annotion_string_from_boxes(boxes)
            labels_dir = os.path.join(self.inference_folder, "processed_images")
            os.makedirs(labels_dir, exist_ok=True)
            print("Created labels folder.")
            annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(annotation_path, "w") as f:
                f.write(annotation_str)
