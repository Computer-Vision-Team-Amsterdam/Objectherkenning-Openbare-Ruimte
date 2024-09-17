from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (
    SilverDetectionMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (
    SilverFrameMetadataManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (
    unix_to_yyyy_mm_dd,
)


class SilverMetadataAggregator:
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.detections = SilverDetectionMetadataManager(spark, catalog, schema)
        self.frames = SilverFrameMetadataManager(spark, catalog, schema)

    def get_image_upload_path_from_detection_id(
        self, detection_id: int, device_id: str
    ) -> str:
        """
        Fetches the image name based on the detection_id, retrieves the gps_date from the frame metadata,
        and constructs the path for uploading the image.
        """
        image_basename = self.detections.get_image_name_from_detection_id(detection_id)
        gps_internal_timestamp = self.frames.get_gps_internal_timestamp_from_image_name(
            image_basename
        )
        date_of_image_upload = unix_to_yyyy_mm_dd(unix_timestamp=gps_internal_timestamp)
        image_upload_path = (
            f"/Volumes/{self.catalog}/default/landingzone/{device_id}/images/"
            f"{date_of_image_upload}/{image_basename}"
        )

        return image_upload_path
