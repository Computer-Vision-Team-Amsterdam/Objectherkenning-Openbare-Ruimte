from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameMetadataManager(TableManager):
    table_name: str = "silver_frame_metadata"
    id_column: str = "frame_id"

    @classmethod
    def get_gps_timestamp_from_image_name(cls, image_name: str) -> str:
        fetch_date_of_image_upload_query = f"""
            SELECT gps_timestamp
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE image_name = '{image_name}'
        """  # nosec
        date_of_image_upload_df = TableManager.spark.sql(
            fetch_date_of_image_upload_query
        )
        date_of_image_upload_dmy = date_of_image_upload_df.collect()[0]["gps_timestamp"]
        return date_of_image_upload_dmy


class SilverFrameMetadataQuarantineManager(TableManager):
    table_name: str = "silver_frame_metadata_quarantine"
    id_column: str = "frame_id"
