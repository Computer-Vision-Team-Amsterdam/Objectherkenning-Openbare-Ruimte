from pyspark.sql.functions import col, date_format  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)

MAX_GPS_DELAY = 5  # in seconds


class BronzeFrameMetadataManager(TableManager):
    table_name: str = "bronze_frame_metadata"

    @classmethod
    def filter_valid_metadata(cls):
        """
        Filters the valid frame metadata based on the conditions that
        gps_lat and gps_lon are not null and not zero.

        Returns:
        -------
        DataFrame
            The DataFrame containing valid frame metadata.
        """

        bronze_frame_metadata = cls.load_pending_rows_from_table()
        valid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNotNull())
            & (col("gps_lat") != 0)
            & (col("gps_lon").isNotNull())
            & (col("gps_lon") != 0)
        )

        valid_metadata = valid_metadata.withColumn(
            "gps_delay",
            (
                col("pylon0_frame_timestamp").cast("long")
                - col("gps_internal_timestamp").cast("long")
            ),
        )

        valid_metadata = valid_metadata.filter(col("gps_delay") <= MAX_GPS_DELAY)
        valid_metadata = valid_metadata.drop("gps_delay")

        print(f"Filtered valid metadata with {valid_metadata.count()} rows.")
        return valid_metadata

    @classmethod
    def filter_invalid_metadata(cls):
        """
        Filters the invalid frame metadata where gps_lat or gps_lon are null or zero.
        Returns:
        -------
        DataFrame
            The DataFrame containing invalid frame metadata.
        """
        bronze_frame_metadata = cls.load_pending_rows_from_table()
        invalid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNull())
            | (col("gps_lat") == 0)
            | (col("gps_lon").isNull())
            | (col("gps_lon") == 0)
        )
        invalid_metadata = invalid_metadata.withColumn(
            "gps_delay",
            (
                col("pylon0_frame_timestamp").cast("long")
                - col("gps_internal_timestamp").cast("long")
            ),
        )

        invalid_metadata = invalid_metadata.filter(col("gps_delay") > MAX_GPS_DELAY)
        invalid_metadata = invalid_metadata.drop("gps_delay")

        print(f"Filtered invalid metadata with {invalid_metadata.count()} rows.")
        return invalid_metadata

    @classmethod
    def get_all_image_names_current_run(cls, job_date: str):
        return (
            cls.get_table()
            .filter((date_format(col("processed_at"), "yyyy-MM-dd") == job_date))
            .select("image_name")
            .rdd.flatMap(lambda x: x)
            .collect()
        )

    @classmethod
    def get_gps_internal_timestamp_of_current_run(cls, job_date: str):
        return (
            cls.get_table()
            .filter((date_format(col("processed_at"), "yyyy-MM-dd") == job_date))
            .select("gps_internal_timestamp")
            .first()[0]
        )
