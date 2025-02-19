from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverDetectionMetadataManager(TableManager):
    table_name: str = "silver_detection_metadata"

    @classmethod
    def get_image_name_from_detection_id(cls, detection_id: int) -> str:
        """
        Fetches the image name corresponding to a specific detection ID from the silver_detection_metadata table.

        Parameters:
        ----------
        detection_id: int
            The ID of the detection for which to retrieve the image name.

        Returns:
        -------
        str: The image name corresponding to the detection ID.
        """
        fetch_image_name_query = f"""
            SELECT image_name
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE id = {detection_id}
        """  # nosec
        image_name_result_df = TableManager.spark.sql(fetch_image_name_query)

        image_name = image_name_result_df.collect()[0]["image_name"]
        return image_name


class SilverDetectionMetadataQuarantineManager(TableManager):
    table_name: str = "silver_detection_metadata_quarantine_bcv_1576"
