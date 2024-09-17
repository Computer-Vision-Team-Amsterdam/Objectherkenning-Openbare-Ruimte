from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameMetadataManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "silver_frame_metadata",
    ):
        super().__init__(spark, catalog, schema, table_name)

    def get_gps_internal_timestamp_from_image_name(self, image_name: str) -> str:
        fetch_date_of_image_upload_query = f"""
            SELECT gps_internal_timestamp
            FROM {self.catalog}.{self.schema}.{self.table_name}
            WHERE image_name = '{image_name}'
        """  # nosec
        date_of_image_upload_df = self.spark.sql(fetch_date_of_image_upload_query)
        date_of_image_upload_dmy = date_of_image_upload_df.collect()[0][
            "gps_internal_timestamp"
        ]
        return date_of_image_upload_dmy


class SilverFrameMetadataQuarantine(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "silver_frame_metadata_quarantine",
    ):
        super().__init__(spark, catalog, schema, table_name)
