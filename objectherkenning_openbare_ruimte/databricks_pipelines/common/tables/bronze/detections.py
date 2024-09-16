import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeDetectionMetadata(TableManager):
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        super().__init__(spark, catalog, schema)
        self.table_name = "bronze_detection_metadata"

    def get_valid_metadata(self, silver_frame_metadata):
        bronze_detection_metadata = self.load_pending_rows_from_table(self.table_name)
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata = self.load_pending_rows_from_table(
            silver_frame_metadata.ta
        )
        silver_frame_metadata = silver_frame_metadata.alias("silver_frame")
        valid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata,
                F.col("bronze_detection.image_name")
                == F.col("silver_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )
        print("Processed valid detection metadata.")

        return valid_metadata

    def get_invalid_metadata(self, silver_frame_metadata_quarantine):
        bronze_detection_metadata = self.load_pending_rows_from_table(self.table_name)
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_quarantine = silver_frame_metadata_quarantine.alias(
            "quarantine_frame"
        )
        invalid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_quarantine,
                F.col("bronze_detection.image_name")
                == F.col("quarantine_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )

        print("Processed invalid detection metadata.")

        return invalid_metadata
