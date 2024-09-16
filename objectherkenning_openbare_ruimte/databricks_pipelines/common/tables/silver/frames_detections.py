from datetime import datetime

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (
    SilverDetectionMetadata,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (
    SilverFrameMetadata,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameAndDetectionMetadata(TableManager):
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        super().__init__(spark, catalog, schema)
        self.detections = SilverDetectionMetadata(spark, catalog, schema)
        self.frames = SilverFrameMetadata(spark, catalog, schema)

    def get_image_upload_path_from_detection_id(
        self, detection_id: int, device_id: str
    ) -> str:
        """
        Fetches the image name based on the detection_id, retrieves the gps_date from the frame metadata,
        and constructs the path for uploading the image.
        """
        image_basename = self.detections.get_image_name_from_detection_id(detection_id)

        gps_date_dmy = self.frames.get_gps_date_from_image_name(image_basename)

        date_obj = datetime.strptime(gps_date_dmy, "%d/%m/%Y")

        gps_date_ymd = date_obj.strftime("%Y-%m-%d")

        image_upload_path = (
            f"/Volumes/{self.catalog}/default/landingzone/{device_id}/images/"
            f"{gps_date_ymd}/{image_basename}"
        )

        return image_upload_path
