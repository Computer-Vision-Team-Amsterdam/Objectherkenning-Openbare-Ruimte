from pyspark.sql.functions import col, date_format

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameMetadataManager(TableManager):
    table_name: str = "silver_frame_metadata"
    id_column: str = "frame_id"

    @classmethod
    def get_upload_date_from_frame_id(cls, frame_id: int) -> str:
        fetch_gps_timestamp_query = f"""
            SELECT gps_timestamp
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE {cls.id_column} = {frame_id}
        """  # nosec
        gps_timestamp_df = TableManager.spark.sql(fetch_gps_timestamp_query)
        upload_date = date_format(
            col(gps_timestamp_df.collect()[0]["gps_timestamp"]), "yyyy-MM-dd"
        ).cast("str")
        return upload_date


class SilverFrameMetadataQuarantineManager(TableManager):
    table_name: str = "silver_frame_metadata_quarantine"
    id_column: str = "frame_id"
