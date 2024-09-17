import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeDetectionMetadataManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "bronze_detection_metadata",
    ):
        super().__init__(spark, catalog, schema, table_name)

    def filter_valid_metadata(self, silver_frame_metadata_df):
        bronze_detection_metadata = self.load_pending_rows_from_table(self.table_name)
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_df = self.load_pending_rows_from_table(
            silver_frame_metadata_df.ta
        )
        silver_frame_metadata_df = silver_frame_metadata_df.alias("silver_frame")
        valid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_df,
                F.col("bronze_detection.image_name")
                == F.col("silver_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )
        print("Processed valid detection metadata.")

        return valid_metadata

    def filter_invalid_metadata(self, silver_frame_metadata_quarantine_df):
        bronze_detection_metadata = self.load_pending_rows_from_table(self.table_name)
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_quarantine_df = silver_frame_metadata_quarantine_df.alias(
            "quarantine_frame"
        )
        invalid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_quarantine_df,
                F.col("bronze_detection.image_name")
                == F.col("quarantine_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )

        print("Processed invalid detection metadata.")

        return invalid_metadata
