# import logging
import os
import secrets
from collections import defaultdict
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
        self.pretrained_model_path = pretrained_model_path
        self.inference_params = inference_params
        print(f"Inference_params: {self.inference_params}")
        print(f"Pretrained_model_path: {self.pretrained_model_path}")
        print(f"Yolo model: {self.model_name}")
        print(f"Project_path: {self.inference_folder}")

        self.model = YOLO(model=self.pretrained_model_path, task="detect")
        self.roi = self.mapx = self.mapy = None
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.tracking_flag = tracking_flag

        # Predefined colors for 5 categories
        predefined_colors = {
            0: (0, 0, 255),  # Blue
            1: (0, 255, 0),  # Green
            2: (255, 0, 0),  # Red
            3: (255, 255, 0),  # Cyan
            4: (255, 0, 255),  # Magenta
        }

        # Initialize the category_colors dictionary with predefined colors and add random colors for new categories
        self.category_colors = defaultdict(
            lambda: (
                secrets.randbelow(256),
                secrets.randbelow(256),
                secrets.randbelow(256),
            ),
            predefined_colors,
        )

    def run_pipeline_prelabeling(self):
        """
        Runs the detection pipeline for pre-labeling or evaluation:
            - find the images to detect;
            - detects everything.
        If tracking_flag = True, it also tracks the detected objects.
        """
        print(f"Running detection pipeline (prelabeling) on {self.images_folder}..")
        videos_and_frames = self._find_image_paths_and_group_by_videoname(
            root_folder=self.images_folder
        )
        print(
            f"Number of images to detect: {sum(len(frames) for frames in videos_and_frames.values())}"
        )
        if self.tracking_flag:
            print("Detecting objects with tracker...")

            self.model = YOLO(model=self.pretrained_model_path, task="track")
            self._track_objects(videos_and_frames=videos_and_frames)
        else:
            print("Detecting objects for prelabeling...")
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
        if self.tracking_flag:
            print("Detecting target classes with tracker...")
            self._track_target_classes(videos_and_frames=videos_and_frames)
        else:
            print("Detecting target classes...")
            self._detect_target_classes(videos_and_frames=videos_and_frames)

    def _detect_target_classes(self, videos_and_frames: Dict[str, List[str]]):
        batch_size = 64
        results = self._process_batches(
            self.model, videos_and_frames, batch_size=batch_size, is_prelabeling=False
        )
        print(results)

    def _detect_objects(self, videos_and_frames: Dict[str, List[str]]):
        batch_size = 1
        results = self._process_batches(
            self.model, videos_and_frames, batch_size=batch_size, is_prelabeling=True
        )
        print(results)

    def _track_target_classes(self, videos_and_frames: Dict[str, List[str]]):
        all_image_paths = [
            path for paths in videos_and_frames.values() for path in paths
        ]
        results = []
        for image_path in all_image_paths:
            # result = self.model(image_path, **self.inference_params)
            result = self.model(
                image_path,
                persist=self.inference_params["persist"],
                tracker=self.inference_params["tracker"],
                save=self.inference_params["save"],
                save_txt=self.inference_params["save_txt"],
                save_conf=self.inference_params["save_conf"],
                conf=self.inference_params["conf"],
                project=self.inference_folder,
                imgsz=self.inference_params["imgsz"],
            )
            results.append(result)
        self._process_results_tracking(results)

    def _track_objects(self, videos_and_frames: Dict[str, List[str]]):
        all_image_paths = [
            path for paths in videos_and_frames.values() for path in paths
        ]
        results = []
        print(self.inference_params)
        for image_path in all_image_paths:
            # result = self.model(image_path, **self.inference_params)
            result = self.model(
                image_path,
                persist=self.inference_params["persist"],
                tracker=self.inference_params["tracker"],
                save=self.inference_params["save"],
                save_txt=self.inference_params["save_txt"],
                save_conf=self.inference_params["save_conf"],
                conf=self.inference_params["conf"],
                project=self.inference_folder,
                imgsz=self.inference_params["imgsz"],
            )
            results.append(result)
        self._process_results_tracking_prelabeling(results)

    @staticmethod
    def _find_image_paths_and_group_by_videoname(root_folder):
        videos_and_frames = {}
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    try:
                        video_name, _ = os.path.basename(filename).rsplit("_", 1)
                    except ValueError as e:
                        print(f"=== ValueError: {e} ===")
                        print(f"Filename: {filename}")
                        video_name = os.path.basename(filename)
                        print(f"Video name: {video_name}")
                    image_path = os.path.join(foldername, filename)
                    if video_name not in videos_and_frames:
                        videos_and_frames[video_name] = [image_path]
                    else:
                        videos_and_frames[video_name].append(image_path)
        return videos_and_frames

    def _process_batches(
        self, model, videos_and_frames, batch_size, is_prelabeling=False
    ):
        for video_name, images_paths in videos_and_frames.items():
            self.inference_params["source"] = images_paths
            processed_images = 0
            for i in range(0, len(images_paths), batch_size):
                try:
                    batch_images = [
                        image_path for image_path in images_paths[i : i + batch_size]
                    ]
                    # batch_images = self._defisheye(batch_images)
                    self.inference_params["source"] = batch_images
                    self.inference_params["name"] = video_name + "_batch_"
                    batch_results = model(**self.inference_params)
                    # print(f"Result YOLO: {batch_results}")
                    torch.cuda.empty_cache()  # Clear unused memory
                    if is_prelabeling:
                        self._process_results_objects(batch_results)
                        if self.tracking_flag:
                            self._process_results_tracking_prelabeling(batch_results)
                    else:
                        self._process_results_target_classes(batch_results)
                        if self.tracking_flag:
                            self._process_results_tracking(batch_results)
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

    def _process_results_target_classes(self, model_results: Results):

        for r in model_results:
            result = r.cpu()
            boxes = result.boxes.numpy()
            image_path = result.path  # Path of the input image
            image_name = os.path.basename(image_path)

            print(f"=== IMAGE NAME: {image_name} ===")

            # print(f"Target classes: {self.target_classes}")
            # print(f"Boxes classes: {boxes.cls}")
            # print(f"Boxes: {boxes}")
            target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
            if len(target_idxs) == 0:  # Nothing to do!
                print("No target classes detected.")
                continue

            image = result.orig_img.copy()
            sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

            # Get categories for each bounding box
            categories = [int(box.cls) for box in boxes]

            # print(f'=== SAVING IMAGE: {image_name} ===')
            print(f"=== CATEGORIES: {categories} ===")

            # Blur sensitive data
            sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
            image = blurring_tools.blur_inside_boxes(image, sensitive_bounding_boxes)

            # Draw annotation boxes
            target_bounding_boxes = boxes[target_idxs].xyxy
            target_categories = [int(box.cls) for box in boxes[target_idxs]]
            image = blurring_tools.draw_bounding_boxes(
                image, target_bounding_boxes, target_categories, self.category_colors
            )

            # Save image
            images_dir = os.path.join(self.inference_folder, "processed_images")
            # folder_path = os.path.join(
            #    self.inference_folder, os.path.basename(os.path.dirname(result.path))
            # )
            os.makedirs(images_dir, exist_ok=True)
            save_path = os.path.join(images_dir, image_name)
            cv2.imwrite(save_path, image)
            # print("Saved image.")

            # Save annotation
            annotation_str = self._get_annotion_string_from_boxes(boxes[target_idxs])
            labels_dir = os.path.join(self.inference_folder, "processed_labels")
            os.makedirs(labels_dir, exist_ok=True)
            annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(annotation_path, "w") as f:
                f.write(annotation_str)

            print("=== SAVED IMAGE ===")

    def _process_results_objects(self, model_results: Results):

        for r in model_results:
            result = r.cpu()
            boxes = result.boxes.numpy()
            image = result.orig_img.copy()

            # Get categories for each bounding box
            categories = [int(box.cls) for box in boxes]

            # Draw annotation boxes
            target_bounding_boxes = boxes.xyxy
            image = blurring_tools.draw_bounding_boxes(
                image, target_bounding_boxes, categories, self.category_colors
            )

            # Save image
            images_dir = os.path.join(
                self.inference_folder, "processed_images_prelabeling"
            )
            # folder_path = os.path.join(
            #    self.inference_folder, os.path.basename(os.path.dirname(result.path))
            # )
            os.makedirs(images_dir, exist_ok=True)
            image_path = result.path  # Path of the input image
            image_name = os.path.basename(image_path)
            save_path = os.path.join(images_dir, image_name)
            cv2.imwrite(save_path, image)
            print("Saved image.")

            # Save annotation
            annotation_str = self._get_annotion_string_from_boxes(boxes)
            labels_dir = os.path.join(
                self.inference_folder, "processed_labels_prelabeling"
            )
            os.makedirs(labels_dir, exist_ok=True)
            annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
            with open(annotation_path, "w") as f:
                f.write(annotation_str)

    def _process_results_tracking(self, model_results: List[Results]):

        for results in model_results:
            for r in results:
                result = r.cpu()
                boxes = result.boxes.numpy()
                image_path = result.path  # Path of the input image
                image_name = os.path.basename(image_path)

                print(f"=== IMAGE NAME: {image_name} ===")

                target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
                if len(target_idxs) == 0:  # Nothing to do!
                    print("No target classes detected.")
                    continue

                image = result.orig_img.copy()
                sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

                # Get categories for each bounding box
                categories = [int(box.cls) for box in boxes]

                # print(f'=== SAVING IMAGE: {image_name} ===')
                print(f"=== CATEGORIES: {categories} ===")

                # Blur sensitive data
                sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
                image = blurring_tools.blur_inside_boxes(
                    image, sensitive_bounding_boxes
                )

                # Draw annotation boxes
                target_bounding_boxes = boxes[target_idxs].xyxy
                tracking_ids = [
                    int(box.id.squeeze()) if box.is_track else -1
                    for box in boxes[target_idxs]
                ]
                target_categories = [int(box.cls) for box in boxes[target_idxs]]
                image = blurring_tools.draw_bounding_boxes(
                    image,
                    target_bounding_boxes,
                    target_categories,
                    self.category_colors,
                    tracking_ids=tracking_ids,
                )

                # Save image
                images_dir = os.path.join(
                    self.inference_folder, "processed_images_tracking"
                )
                # folder_path = os.path.join(
                #    self.inference_folder, os.path.basename(os.path.dirname(result.path))
                # )
                os.makedirs(images_dir, exist_ok=True)
                image_path = result.path  # Path of the input image
                image_name = os.path.basename(image_path)
                save_path = os.path.join(images_dir, image_name)
                cv2.imwrite(save_path, image)
                # print("Saved image.")

                # Save annotation
                annotation_str = self._get_annotion_string_from_boxes(
                    boxes[target_idxs]
                )
                labels_dir = os.path.join(
                    self.inference_folder, "processed_labels_tracking"
                )
                os.makedirs(labels_dir, exist_ok=True)
                annotation_path = os.path.join(
                    labels_dir, f"{os.path.splitext(image_name)[0]}.txt"
                )
                with open(annotation_path, "w") as f:
                    f.write(annotation_str)

                print("=== SAVED IMAGE ===")

        print("Tracking complete. Results saved.")

    def _process_results_tracking_prelabeling(self, model_results: List[Results]):

        labels_dir = os.path.join(
            self.inference_folder, "processed_labels_tracking_prelabeling"
        )
        os.makedirs(labels_dir, exist_ok=True)

        for results in model_results:
            for r in results:
                result = r.cpu()
                boxes = result.boxes.numpy()
                image_path = result.path  # Path of the input image
                image_name = os.path.basename(image_path)

                target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]

                # Save annotation
                annotation_str = self._get_annotion_string_from_boxes(
                    boxes[target_idxs]
                )
                annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
                with open(annotation_path, "w") as f:
                    f.write(annotation_str)

        print("Tracking complete. Results saved.")
