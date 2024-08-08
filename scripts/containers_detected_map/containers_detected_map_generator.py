import base64
import glob
import os
import random
from typing import List

import folium
import pandas as pd


class ContainersDetectedMapGenerator:
    def __init__(
        self,
        date_of_the_ride: str,
        confidence_threshold: float = 0.7,
        size_of_detection_threshold: float = 0.003,
    ):
        """The data needs to be located in the same folder with the following structure:
        - frame_metadata/date
        - full_frame_metadata/date
        - detection_metadata/date

        Parameters
        ----------
        date_of_the_ride : str
            Date of the ride that it's also the name of the subfolder date.
        confidence_threshold : float, optional
            Model confidence threshold, by default 0.7
        size_of_detection_threshold : float, optional
            Minimum size of detection, by default 0.003
        """
        self.folder_name = date_of_the_ride
        self.frame_metadata_folder = f"frame_metadata/{self.folder_name}"
        self.full_frame_metadata_folder = f"full_frame_metadata/{self.folder_name}"
        self.detection_metadata_folder = f"detection_metadata/{self.folder_name}"
        self.images_folder = f"images/{self.folder_name}"
        self.confidence_threshold = confidence_threshold
        self.size_of_detection_threshold = size_of_detection_threshold

    def create_and_store_map(self):
        detections_map = self._create_map()
        self._save_map(detections_map)

    def _create_map(self):
        frame_csv_files, full_frame_csv_files, detection_csv_files = (
            self._find_all_csv_files()
        )
        detection_data = pd.concat([pd.read_csv(file) for file in detection_csv_files])

        initial_location = None
        detections_map = None
        for file_path in frame_csv_files:
            frame_data = pd.read_csv(file_path)
            full_frame_path = self._get_full_metadata_file_path(
                full_frame_csv_files=full_frame_csv_files,
                frame_metadata_file_path=file_path,
            )

            if (
                "gps_lat" not in frame_data.columns
                or "gps_lon" not in frame_data.columns
                or "image_name" not in frame_data.columns
                or full_frame_path is None
            ):
                print(
                    f"Skipping {file_path}: Missing 'gps_lat', 'gps_lon', or 'image_name' columns."
                )
                continue

            full_frame_data = pd.read_csv(full_frame_path)
            locations = list(
                zip(full_frame_data["gps_lat"], full_frame_data["gps_lon"])
            )
            if initial_location is None:
                initial_location = locations[0]
                detections_map = folium.Map(location=initial_location, zoom_start=18)

            detections_map = self._add_bike_path_to_map(locations, detections_map)
            detections_map = self._add_dots_and_images_to_map(
                frame_data, detection_data, detections_map
            )
        return detections_map

    def _save_map(self, detections_map):
        if detections_map is None:
            raise ValueError("No valid frame metadata CSV files found.")

        output_map = f"{self.folder_name}_{self.confidence_threshold}_{self.size_of_detection_threshold}.html"
        detections_map.save(output_map)
        print(f"Map has been saved to {output_map}")

    def _add_dots_and_images_to_map(self, frame_data, detection_data, detections_map):
        merged_data = pd.merge(frame_data, detection_data, on="image_name")
        for _, row in merged_data.iterrows():
            size_of_detection = row["width"] * row["height"]
            if (
                row["confidence"] > self.confidence_threshold
                and size_of_detection > self.size_of_detection_threshold
            ):
                html = f"""
                <b>Image Name:</b> {row['image_name']}<br>
                <b>Confidence:</b> {row['confidence']}<br>
                """

                image_path = os.path.join(self.images_folder, f"{row['image_name']}")
                if os.path.isfile(image_path):
                    encoded = base64.b64encode(open(image_path, "rb").read())
                    html_image = '<img src="data:image/jpg;base64,{}" width="600" height="400">'.format
                    html += html_image(encoded.decode("UTF-8"))

                iframe = folium.IFrame(html, width=800, height=500)
                popup = folium.Popup(iframe, min_width=450, max_width=900)
                folium.CircleMarker(
                    location=[row["gps_lat"], row["gps_lon"]],
                    radius=5,  # Size of the dot
                    color="red",  # Color of the dot
                    fill=True,
                    fill_color="red",
                    popup=popup,
                ).add_to(detections_map)
        return detections_map

    def _add_bike_path_to_map(self, locations, detections_map):
        color = self._get_random_color()
        folium.PolyLine(locations, color=color, weight=7, opacity=0.9).add_to(
            detections_map
        )
        return detections_map

    @staticmethod
    def _get_full_metadata_file_path(
        full_frame_csv_files: List[str], frame_metadata_file_path: str
    ):
        for full_file_path in full_frame_csv_files:
            if frame_metadata_file_path in full_file_path:
                return full_file_path
        return None

    def _find_all_csv_files(self):
        frame_csv_files = self._get_csvs_in_folder(self.frame_metadata_folder)
        full_frame_csv_files = self._get_csvs_in_folder(self.full_frame_metadata_folder)
        detection_csv_files = self._get_csvs_in_folder(self.detection_metadata_folder)

        if not full_frame_csv_files:
            raise ValueError(
                "No full frame metadata CSV files found in the specified folder."
            )
        if not frame_csv_files:
            raise ValueError(
                "No frame metadata CSV files found in the specified folder."
            )
        if not detection_csv_files:
            raise ValueError(
                "No detection metadata CSV files found in the specified folder."
            )

        return frame_csv_files, full_frame_csv_files, detection_csv_files

    @staticmethod
    def _get_random_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))  # nosec

    @staticmethod
    def _get_csvs_in_folder(folder: str):
        return glob.glob(os.path.join(folder, "*.csv"))
