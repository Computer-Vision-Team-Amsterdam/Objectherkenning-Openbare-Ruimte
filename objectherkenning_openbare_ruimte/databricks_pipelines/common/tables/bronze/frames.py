from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeFrameMetadata(TableManager):
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        super().__init__(spark, catalog, schema)
        self.table_name = "bronze_frame_metadata"

    def get_valid_metadata(self):
        """
        Filters the valid frame metadata based on the conditions that
        gps_lat and gps_lon are not null and not zero.

        Returns:
        -------
        DataFrame
            The DataFrame containing valid frame metadata.
        """

        bronze_frame_metadata = self.load_pending_rows_from_table(self.table_name)
        valid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNotNull())
            & (col("gps_lat") != 0)
            & (col("gps_lon").isNotNull())
            & (col("gps_lon") != 0)
        )
        print(f"Filtered valid metadata with {valid_metadata.count()} rows.")
        return valid_metadata

    def get_invalid_metadata(self):
        """
        Filters the invalid frame metadata where gps_lat or gps_lon are null or zero.
        Returns:
        -------
        DataFrame
            The DataFrame containing invalid frame metadata.
        """
        bronze_frame_metadata = self.load_pending_rows_from_table(self.table_name)
        invalid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNull())
            | (col("gps_lat") == 0)
            | (col("gps_lon").isNull())
            | (col("gps_lon") == 0)
        )
        print(f"Filtered invalid metadata with {invalid_metadata.count()} rows.")
        return invalid_metadata
