from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from databricks.sdk.runtime import dbutils
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    delete_file_or_folder,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.aggregators import (
    SilverMetadataAggregator,
)

IMAGE_FORMATS = (".jpg", ".jpeg", ".png")


class DeleteImagesStep:
    """
    Delete image files from the landing zone following the data retention policy.
    """

    dry_run = True

    def __init__(
        self,
        spark_session: SparkSession,
        catalog: str,
        schema: str,
        settings: dict[str, Any],
    ):
        self.spark_session = spark_session
        self.catalog = catalog
        self.schema = schema
        self.device_id = settings["device_id"]
        self.min_score = settings["job_config"]["min_score"]
        self.retention_weeks = settings["job_config"]["retention_weeks"]
        self.detection_date = settings["job_config"].get("detection_date", None)

    def run(self) -> None:
        """
        Run the delete images step:
        - Scan input folder for image files (A);
        - Get all processed images from silver tables (B);
        - Select images to keep based on retention policy (C);
        - Delete images from A that are in B but not in C.

        Will also delete visualizations that are older than the retention policy.
        """
        print("\n** Deleting images **")
        root_image_folder = (
            f"/Volumes/{self.catalog}/default/landingzone/{self.device_id}/images/"
        )
        folders_and_images = self.get_folders_and_images(
            root_image_folder, self.detection_date
        )
        print("")

        silverMetadataAggregator = SilverMetadataAggregator(
            spark_session=self.spark_session,
            catalog=self.catalog,
            schema=self.schema,
        )

        processed_detections = silverMetadataAggregator.get_joined_processed_metadata(
            self.detection_date
        )
        detections_to_keep = self.filter_detections_to_keep(processed_detections)

        for date_folder in sorted(folders_and_images.keys()):
            print(f"\nChecking images for date {date_folder}")

            delete_candidate_image_names = set(
                self.get_image_names_at_date(processed_detections, date_folder)
            )
            to_keep_image_names = set(
                self.get_image_names_at_date(detections_to_keep, date_folder)
            )
            to_delete_image_names = delete_candidate_image_names - to_keep_image_names

            image_files_to_keep = [
                file
                for file in folders_and_images[date_folder]
                if file.name in to_keep_image_names
            ]
            image_files_to_delete = [
                file
                for file in folders_and_images[date_folder]
                if file.name in to_delete_image_names
            ]

            print(f" - {len(folders_and_images[date_folder])} images found")
            print(f" - {len(image_files_to_keep)} images to keep")
            print(f" - {len(image_files_to_delete)} images to delete")

            successful_deletions = 0
            for file in image_files_to_delete:
                if self.dry_run:
                    print(f"   Deleting {file.path}...")
                else:
                    if delete_file_or_folder(databricks_volume_full_path=file.path):
                        successful_deletions += 1
            print(f" - {successful_deletions} images successfully deleted.")

        print("\n** Deleting empty subfolders **")
        self.delete_empty_subfolders(root_image_folder)

        print("\n** Deleting Visualizations **")
        root_visualizations_folder = f"/Volumes/{self.catalog}/default/landingzone/{self.device_id}/visualizations/"
        self.delete_visualizations(root_visualizations_folder)

        print("\n** All done! **")

    def filter_detections_to_keep(self, detections: DataFrame) -> DataFrame:
        """
        Filter detections following the data retention policy for images:
        - Delete all images that are more than retention_weeks old;
        - Keep images that are less than retention_weeks old and have a score above the score_threshold.
        """
        date_cutoff = (datetime.now() - timedelta(weeks=self.retention_weeks)).date()
        print(
            f"Filtering detections using cutoff date {date_cutoff} and score threshold {self.min_score}..."
        )
        return detections.filter(
            (col("detection_date") >= date_cutoff) & (col("score") >= self.min_score)
        )

    def delete_empty_subfolders(self, root_folder: str) -> None:
        """Delete empty subfolders."""
        print(f"Scanning {root_folder} for empty subfolders...")
        subfolder_list = [
            folder for folder in dbutils.fs.ls(root_folder) if folder.isDir()
        ]
        for subfolder in subfolder_list:
            contents = dbutils.fs.ls(subfolder.path)
            if len(contents) == 0:
                print(f"Deleting empty folder: {subfolder.path}")
                if not self.dry_run:
                    delete_file_or_folder(subfolder.path)

    def delete_visualizations(self, root_folder: str) -> None:
        """
        Delete visualization folders following the retention policy for visualizations:
        - Delete all subfolders that are more than retention_weeks old.
        """
        date_cutoff = (datetime.now() - timedelta(weeks=self.retention_weeks)).date()
        print(f"Scanning {root_folder} with cutoff date {date_cutoff}...")
        subfolder_deletion_list = [
            folder
            for folder in dbutils.fs.ls(root_folder)
            if folder.isDir()
            and (date.fromisoformat(folder.name.strip("/")) < date_cutoff)
        ]
        for subfolder in subfolder_deletion_list:
            print(
                f"Deleting visualizations folder (including contents): {subfolder.path}"
            )
            if not self.dry_run:
                delete_file_or_folder(subfolder.path, recurse=True)

    @classmethod
    def get_image_names_at_date(
        cls, detections: DataFrame, detection_date: str
    ) -> List[str]:
        """Returns the set of unique images names at the given detection date."""
        return (
            detections.filter(col("detection_date") == detection_date)
            .select("image_name")
            .rdd.flatMap(lambda x: x)
            .collect()
        )

    @classmethod
    def get_folders_and_images(
        cls, root_folder: str, detection_date: Optional[str] = None
    ) -> Dict[str, List[Any]]:
        """
        Scan the root_folder and return a dict with each subfolder as key, and
        all image files within that subfolder as values. Optionally limits the
        result to a given detection_date.
        """
        folders_and_images: Dict[str, List[Any]] = dict()

        if detection_date is not None:
            print(f"Scanning {root_folder} for date {detection_date}...")
        else:
            print(f"Scanning {root_folder}...")

        subfolder_list = [
            folder.name.strip("/")
            for folder in dbutils.fs.ls(root_folder)
            if folder.isDir()
        ]

        if detection_date is not None:
            subfolder_list = list(set(subfolder_list).intersection(set(detection_date)))
            if len(subfolder_list) == 0:
                print(f"No subfolder found for {detection_date}")
                return folders_and_images

        for subfolder in subfolder_list:
            folders_and_images[subfolder] = [
                file
                for file in dbutils.fs.ls(f"{root_folder}/{subfolder}/")
                if file.name.lower().endswith(IMAGE_FORMATS)
            ]

        n_folders = len(folders_and_images.keys())
        n_images = sum(
            len(folders_and_images[folder]) for folder in folders_and_images.keys()
        )
        print(f"Found {n_images} images in {n_folders} subfolders")

        return folders_and_images
