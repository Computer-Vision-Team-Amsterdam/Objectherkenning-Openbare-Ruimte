from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, explode, lit, to_timestamp


class JsonFrameDetectionAdapter:
    """
    Reads JSON files from landingzone stream. Provides methods to convert JSON
    metadata to the required formats for bronze_frame_metadata and
    bronze_detection_metadata.
    """

    def __init__(
        self,
        spark_session: SparkSession,
        json_source: str,
        frame_schema_loc: str,
        detection_schema_loc: str,
    ):
        # Read JSON data twice, each with its own schemaLocation
        self.raw_frames = self._read_stream(
            spark_session, json_source, frame_schema_loc
        )
        self.raw_dets = self._read_stream(
            spark_session, json_source, detection_schema_loc
        )

    def _read_stream(
        self, spark_session: SparkSession, json_source: str, schema_location: str
    ) -> DataFrame:
        # `.option("cloudFiles.schemaEvolutionMode", "none")` ignores any new columns in the input data
        # See https://learn.microsoft.com/en-us/azure/databricks/ingestion/cloud-object-storage/auto-loader/schema#evolution

        data = (
            spark_session.readStream.format("cloudFiles")
            .option("multiline", "true")
            .option("cloudFiles.format", "json")
            .option("pathGlobFilter", "*.json")
            .option("cloudFiles.schemaLocation", schema_location)
            .option("cloudFiles.inferColumnTypes", "true")
            .option("ignoreMissingFiles", "true")
            .option("cloudFiles.schemaEvolutionMode", "none")
            .load(json_source)
        )
        return data

    def to_frame_df(self) -> DataFrame:
        """
        Return dataframe matching the bronze_frame_metadata format
        """
        # Produce exactly the columns in bronze_frame_metadata
        return self.raw_frames.select(
            col("image_file_name").alias("image_name"),
            to_timestamp(col("image_file_timestamp")).alias("image_timestamp"),
            col("gps_data.latitude").cast("double").alias("gps_lat"),
            col("gps_data.longitude").cast("double").alias("gps_lon"),
            to_timestamp(col("gps_data.coordinate_time_stamp")).alias("gps_timestamp"),
            col("project.model_name").alias("model_name"),
            col("project.aml_model_version").cast("int").alias("aml_model_version"),
            col("project.project_version").alias("project_version"),
            col("project.customer").alias("customer"),
        ).withColumn("status", lit("Pending").cast("string"))

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
