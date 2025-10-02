from typing import Optional

from pyspark.sql import DataFrame, SparkSession

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

    def get_joined_processed_metadata(self, detection_date: Optional[str]) -> DataFrame:
        """
        Retrieves a join of frames, detections, and enriched detections that are
        not "Pending", optionally limited to a specified date.

        Parameters:
            detection_date (optional): The date (in "yyyy-MM-dd" format) to
            filter the join by.

        Returns:
            A DataFrame with columns [detection_id, score, frame_id, image_name,
            detection_date]. Score is NULL if that detections was not selected
            after enrichment.
        """

        query = """
            SELECT sd.detection_id, sed.score, sed.status, sf.frame_id, sf.image_name, DATE(sf.gps_timestamp) AS detection_date
            FROM :catalog.:schema.silver_frame_metadata AS sf
            LEFT JOIN :catalog.:schema.silver_detection_metadata AS sd ON sf.frame_id = sd.frame_id
            LEFT JOIN :catalog.:schema.silver_enriched_detection_metadata AS sed ON sd.detection_id = sed.detection_id
            WHERE sf.status == "Processed"
            AND (sed.status == "Processed" OR sed.status IS NULL)
        """

        params = {"catalog": self.catalog, "schema": self.schema}

        if detection_date is not None:
            query += """
            AND DATE(sf.gps_timestamp) == :detection_date
            """
            params["detection_date"] = detection_date

        return self.spark_session.sql(query, params)
