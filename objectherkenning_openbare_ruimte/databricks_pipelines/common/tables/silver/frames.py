from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameMetadataManager(TableManager):
    table_name: str = "silver_frame_metadata"

    @staticmethod
    def get_table_name() -> str:
        return SilverFrameMetadataManager.table_name

    @staticmethod
    def get_gps_internal_timestamp_from_image_name(image_name: str) -> str:
        fetch_date_of_image_upload_query = f"""
            SELECT gps_internal_timestamp
            FROM {SilverFrameMetadataManager.catalog}.{SilverFrameMetadataManager.schema}.{SilverFrameMetadataManager.table_name}
            WHERE image_name = '{image_name}'
        """  # nosec
        date_of_image_upload_df = SilverFrameMetadataManager.spark.sql(
            fetch_date_of_image_upload_query
        )
        date_of_image_upload_dmy = date_of_image_upload_df.collect()[0][
            "gps_internal_timestamp"
        ]
        return date_of_image_upload_dmy


class SilverFrameMetadataQuarantineManager(TableManager):
    table_name: str = "silver_frame_metadata_quarantine"

    @staticmethod
    def get_table_name() -> str:
        return SilverFrameMetadataQuarantineManager.table_name
