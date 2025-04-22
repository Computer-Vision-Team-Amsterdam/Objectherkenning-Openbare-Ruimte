from pyspark.sql.functions import col, date_format, explode, lit
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


class JsonFrameDetAdapter:
    json_schema = StructType(
        [
            StructField("frame_number", IntegerType()),
            StructField("record_timestamp", TimestampType()),
            StructField("image_file_timestamp", TimestampType()),
            StructField("image_file_name", StringType()),
            StructField(
                "gps_data",
                StructType(
                    [
                        StructField("latitude", DoubleType()),
                        StructField("longitude", DoubleType()),
                        StructField("altitude", DoubleType()),
                        StructField("coordinate_time_stamp", TimestampType()),
                    ]
                ),
            ),
            StructField(
                "project",
                StructType(
                    [
                        StructField("model_name", StringType()),
                        StructField("aml_model_version", IntegerType()),
                        StructField("project_version", StringType()),
                        StructField("customer", StringType()),
                    ]
                ),
            ),
            StructField(
                "detections",
                ArrayType(
                    StructType(
                        [
                            StructField("object_class", IntegerType()),
                            StructField("confidence", DoubleType()),
                            StructField("tracking_id", IntegerType()),
                            StructField(
                                "boundingBox",
                                StructType(
                                    [
                                        StructField("x_center", DoubleType()),
                                        StructField("y_center", DoubleType()),
                                        StructField("width", DoubleType()),
                                        StructField("height", DoubleType()),
                                    ]
                                ),
                            ),
                        ]
                    )
                ),
            ),
        ]
    )

    def __init__(self, spark, json_source):
        self.spark = spark
        self.json_source = json_source

    def load_raw(self):
        return (
            self.spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "json")
            .option("pathGlobFilter", "*.json")
            .schema(self.json_schema)
            .load(self.json_source)
        )

    def to_frame_df(self, raw):
        # Produce exactly the columns in bronze_frame_metadata
        df = raw.select(
            col("record_timestamp").alias("timestamp"),
            col("frame_number").alias("pylon0_frame_counter"),
            col("image_file_timestamp").alias("pylon0_frame_timestamp"),
            col("gps_data.coordinate_time_stamp").alias("gps_timestamp"),
            col("gps_data.coordinate_time_stamp").alias("gps_internal_timestamp"),
            col("gps_data.latitude").alias("gps_lat"),
            col("gps_data.longitude").alias("gps_lon"),
            col("image_file_name").alias("image_name"),
            col("project.model_name").alias("model_name"),
            col("project.aml_model_version").alias("model_version"),
            col("project.project_version").alias("code_version"),
            date_format(col("gps_data.coordinate_time_stamp"), "yyyy-MM-dd").alias(
                "gps_date"
            ),
            col("gps_data.coordinate_time_stamp").alias("gps_time"),
        )

        # add all the missing columns as NULL/defaults:
        df = (
            df.withColumn("imu_state", lit(None).cast("int"))
            .withColumn("imu_pitch", lit(None).cast("float"))
            .withColumn("imu_roll", lit(None).cast("float"))
            .withColumn("imu_heading", lit(None).cast("float"))
            .withColumn("imu_gx", lit(None).cast("int"))
            .withColumn("imu_gy", lit(None).cast("int"))
            .withColumn("imu_gz", lit(None).cast("int"))
            .withColumn("gps_state", lit(None).cast("int"))
            .withColumn("status", lit("Pending").cast("string"))
        )
        return df

    def to_det_df(self, raw):
        # Produce exactly the columns in bronze_detection_metadata
        exploded = raw.select(
            col("image_file_name").alias("image_name"),
            explode("detections").alias("det"),
        )
        return exploded.select(
            col("image_name"),
            col("det.object_class").alias("object_class"),
            col("det.boundingBox.x_center").alias("x_center"),
            col("det.boundingBox.y_center").alias("y_center"),
            col("det.boundingBox.width").alias("width"),
            col("det.boundingBox.height").alias("height"),
            col("det.confidence").alias("confidence"),
            col("det.tracking_id").alias("tracking_id"),
        ).withColumn("status", lit("Pending"))

    def load(self):
        raw = self.load_raw()
        return self.to_frame_df(raw), self.to_det_df(raw)
