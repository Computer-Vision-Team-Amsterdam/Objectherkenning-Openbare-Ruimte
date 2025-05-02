from typing import Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, date_format, explode, lit, to_timestamp


class JsonFrameDetAdapter:
    """
    Reads JSON files from landingzone stream. Provides methods to convert JSON
    metadata to the required formats for bronze_frame_metadata and
    bronze_detection_metadata.
    """

    def __init__(
        self,
        spark: SparkSession,
        json_source: str,
        frame_schema_loc: str,
        det_schema_loc: str,
    ):
        # Read JSON data twice, each with its own schemaLocation
        self.raw_frames = (
            spark.readStream.format("cloudFiles")
            .option("multiline", "true")
            .option("cloudFiles.format", "json")
            .option("pathGlobFilter", "*.json")
            .option("cloudFiles.schemaLocation", frame_schema_loc)
            .option("cloudFiles.inferColumnTypes", "true")
            .load(json_source)
        )
        self.raw_dets = (
            spark.readStream.format("cloudFiles")
            .option("multiline", "true")
            .option("cloudFiles.format", "json")
            .option("pathGlobFilter", "*.json")
            .option("cloudFiles.schemaLocation", det_schema_loc)
            .option("cloudFiles.inferColumnTypes", "true")
            .load(json_source)
        )

    def get_load_counts(self) -> Tuple[int, int]:
        return (self.raw_frames.count(), self.raw_dets.count())

    def to_frame_df(self) -> DataFrame:
        """
        Return dataframe matching the bronze_frame_metadata format
        """
        # Produce exactly the columns in bronze_frame_metadata
        df = self.raw_frames.select(
            to_timestamp(col("record_timestamp")).cast("double").alias("timestamp"),
            col("frame_number").cast("integer").alias("pylon0_frame_counter"),
            to_timestamp(col("image_file_timestamp"))
            .cast("double")
            .alias("pylon0_frame_timestamp"),
            to_timestamp(col("gps_data.coordinate_time_stamp"))
            .cast("double")
            .alias("gps_timestamp"),
            to_timestamp(col("gps_data.coordinate_time_stamp"))
            .cast("double")
            .alias("gps_internal_timestamp"),
            col("gps_data.latitude").cast("string").alias("gps_lat"),
            col("gps_data.longitude").cast("string").alias("gps_lon"),
            col("image_file_name").alias("image_name"),
            col("project.model_name").alias("model_name"),
            col("project.aml_model_version").cast("int").alias("model_version"),
            col("project.project_version").alias("code_version"),
            date_format(col("gps_data.coordinate_time_stamp"), "yyyy-MM-dd").alias(
                "gps_date"
            ),
            to_timestamp(col("gps_data.coordinate_time_stamp")).alias("gps_time"),
        )

        # add all the missing columns as NULL/defaults:
        df = (
            df.withColumn("imu_state", lit(None).cast("integer"))
            .withColumn("imu_pitch", lit(None).cast("float"))
            .withColumn("imu_roll", lit(None).cast("float"))
            .withColumn("imu_heading", lit(None).cast("float"))
            .withColumn("imu_gx", lit(None).cast("float"))
            .withColumn("imu_gy", lit(None).cast("float"))
            .withColumn("imu_gz", lit(None).cast("float"))
            .withColumn("gps_state", lit(None).cast("integer"))
            .withColumn("status", lit("Pending").cast("string"))
        )
        return df

    def to_det_df(self) -> DataFrame:
        """
        Return dataframe matching the bronze_detection_metadata format
        """
        # Produce exactly the columns in bronze_detection_metadata
        exploded = self.raw_dets.select(
            col("image_file_name").alias("image_name"),
            explode("detections").alias("det"),
        )
        return exploded.select(
            col("image_name"),
            col("det.object_class").cast("integer").alias("object_class"),
            col("det.boundingBox.x_center").cast("float").alias("x_center"),
            col("det.boundingBox.y_center").cast("float").alias("y_center"),
            col("det.boundingBox.width").cast("float").alias("width"),
            col("det.boundingBox.height").cast("float").alias("height"),
            col("det.confidence").cast("float").alias("confidence"),
            col("det.tracking_id").cast("integer").alias("tracking_id"),
        ).withColumn("status", lit("Pending").cast("string"))
