from typing import Tuple

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverDetectionMetadataManager(TableManager):
    table_name: str = "silver_detection_metadata"
    id_column: str = "detection_id"

    @classmethod
    def get_image_name_from_detection_id(cls, detection_id: int) -> str:
        """
        Fetches the image name corresponding to a specific detection ID from the
        silver_detection_metadata table.

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
            WHERE {cls.id_column} = {detection_id}
        """  # nosec
        image_name_result_df = TableManager.spark_session.sql(fetch_image_name_query)

        image_name = image_name_result_df.collect()[0]["image_name"]
        return image_name

    @classmethod
    def get_frame_id_from_detection_id(cls, detection_id: int) -> int:
        """
        Fetches the frame ID corresponding to a specific detection ID from the
        silver_detection_metadata table.

        Parameters:
        ----------
        detection_id: int
            The ID of the detection for which to retrieve the image name.

        Returns:
        -------
        int: The frame ID name corresponding to the detection ID.
        """
        fetch_frame_id_query = f"""
            SELECT frame_id
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE {cls.id_column} = {detection_id}
        """  # nosec
        frame_id_result_df = TableManager.spark_session.sql(fetch_frame_id_query)

        frame_id = frame_id_result_df.collect()[0]["frame_id"]
        return frame_id

    @classmethod
    def get_bounding_box_from_detection_id(
        cls, detection_id: int
    ) -> Tuple[float, float, float, float]:
        """
        Fetches the bounding box corresponding to a specific detection ID from the
        silver_detection_metadata table.

        Parameters:
        ----------
        detection_id: int
            The ID of the detection for which to retrieve the image name.

        Returns:
        -------
        Tuple: The bounding box as (x_center, y_center, width, height)
        """
        fetch_bounding_box_query = f"""
            SELECT x_center, y_center, width, height
            FROM {TableManager.catalog}.{TableManager.schema}.{cls.table_name}
            WHERE {cls.id_column} = {detection_id}
        """  # nosec
        result_df = TableManager.spark_session.sql(fetch_bounding_box_query)

        bounding_box = (
            result_df[["x_center", "y_center", "width", "height"]]
            .rdd.map(tuple)
            .collect()[0]
        )
        return bounding_box


class SilverDetectionMetadataQuarantineManager(TableManager):
    table_name: str = "silver_detection_metadata_quarantine"
    id_column: str = "detection_id"
