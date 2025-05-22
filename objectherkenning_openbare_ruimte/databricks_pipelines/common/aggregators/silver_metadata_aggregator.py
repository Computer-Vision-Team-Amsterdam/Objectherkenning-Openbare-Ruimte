from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (
    SilverDetectionMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (
    SilverFrameMetadataManager,
)


class SilverMetadataAggregator:
    def __init__(self, spark_session: SparkSession, catalog: str, schema: str):
        self.spark_session = spark_session
        self.catalog = catalog
        self.schema = schema

    def get_image_upload_path_from_detection_id(
        self, detection_id: int, device_id: str
    ) -> str:
        """
        Fetches the image name based on the detection_id, retrieves the gps_date from the frame metadata,
        and constructs the path for uploading the image.
        """
        image_basename = (
            SilverDetectionMetadataManager.get_image_name_from_detection_id(
                detection_id
            )
        )
        frame_id = SilverDetectionMetadataManager.get_frame_id_from_detection_id(
            detection_id
        )
        date_of_upload = SilverFrameMetadataManager.get_upload_date_from_frame_id(
            frame_id
        )
        image_upload_path = (
            f"/Volumes/{self.catalog}/default/landingzone/{device_id}/images/"
            f"{date_of_upload}/{image_basename}"
        )

        return image_upload_path
