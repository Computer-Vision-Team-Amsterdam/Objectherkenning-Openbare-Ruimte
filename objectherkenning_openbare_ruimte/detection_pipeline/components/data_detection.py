from typing import List

from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths


class DataDetection:
    def __init__(
        self,
        images_folder: str,
    ):
        """

        Parameters
        ----------
        images_folder
            Folder containing unblurred images.
        """
        self.images_folder = images_folder

    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects containers;
            - deletes the raw images.
        """
        print(f"Running container detection pipeline on {self.images_folder}..")
        images_paths = find_image_paths(root_folder=self.images_folder)
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
        pass
