from typing import Any

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    delete_file,
    get_job_process_time,
    get_landingzone_folder_for_timestamp,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    BronzeFrameMetadataManager,
    SilverDetectionMetadataManager,
    SilverEnrichedDetectionMetadataManager,
)


class DeleteImagesStep:
    """
    Delete image files from the landing zone following the data retention policy.
    """

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

    def run(self):
        # TODO
        # Refactor

        job_process_time = (
            get_job_process_time(
                is_first_pipeline_step=False,
            ),
        )

        job_date = job_process_time.strftime("%Y-%m-%d")

        # TODO
        # - List all image files in all subfolders (date agnostic)
        # - Get all detection_ids that have a score > 0.4
        #   (or some other retention rule, make this a function that can easily be modified later)
        # - Get image names for those detection_ids
        # - Check which images are older than the last processed time in bronze, and are not on the list of images to keep
        # - Delete those

        stlanding_image_folder = get_landingzone_folder_for_timestamp(
            BronzeFrameMetadataManager.get_gps_timestamp_at_date(job_date=job_date)
        )
        image_files_current_run = dbutils.fs.ls(f"/Volumes/{self.catalog}/default/landingzone/{self.device_id}/images/{stlanding_image_folder}/")  # type: ignore[name-defined] # noqa: F821, F405
        print(
            f"{len(image_files_current_run)} images found on {stlanding_image_folder}."
        )

        # Must compare candidates for deletion and images to keep, since one image may have multiple detections.
        delete_candidate_image_names: set = {
            SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
            for d in SilverEnrichedDetectionMetadataManager.get_detection_ids_candidates_for_deletion(
                job_date=job_date, score_threshold=self.min_score
            )
        }
        to_keep_image_names: set = {
            SilverDetectionMetadataManager.get_image_name_from_detection_id(d)
            for d in SilverEnrichedDetectionMetadataManager.get_detection_ids_to_keep_current_run(
                job_date=job_date, score_threshold=self.min_score
            )
        }
        to_delete_image_names = delete_candidate_image_names - to_keep_image_names
        print(f"{len(to_delete_image_names)} images to delete.")

        successful_deletions = 0
        for file in image_files_current_run:
            image_name = file.name

            if image_name in to_delete_image_names:
                print(f"Deleting {image_name}...")
                if delete_file(databricks_volume_full_path=file.path):
                    successful_deletions += 1
        print(f"{successful_deletions} images successfully deleted.")
