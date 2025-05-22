from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverFrameMetadataManager(TableManager):
    table_name: str = "silver_frame_metadata"
    id_column: str = "frame_id"

    @classmethod
    def get_upload_date_from_frame_id(cls, frame_id: int) -> str:
        fetch_gps_timestamp_query = f"""
            SELECT DATE(gps_timestamp) AS upload_date
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE {cls.id_column} = {frame_id}
        """  # nosec
        upload_date_df = TableManager.spark_session.sql(fetch_gps_timestamp_query)
        row = upload_date_df.first()
        if row is None:
            raise ValueError(f"Frame_id {frame_id} not found")
        return str(row["upload_date"])


class SilverFrameMetadataQuarantineManager(TableManager):
    table_name: str = "silver_frame_metadata_quarantine"
    id_column: str = "frame_id"
