import pyspark.sql.functions as F
from pyspark.sql.functions import col


class MetadataHealthChecker:
    def __init__(self, spark, catalog, schema, job_process_time):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.job_process_time = job_process_time

    def load_bronze_metadata(self, table_name):
        df = self.spark.table(f"{self.catalog}.{self.schema}.{table_name}").filter(
            "status = 'Pending'"
        )
        print(
            f"02: Loaded {df.count()} 'Pending' rows from {self.catalog}.{self.schema}.{table_name}."
        )
        return df

    def process_frame_metadata(self, bronze_frame_metadata):

        # Minimal check, we should implement more logic here
        valid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNotNull())
            & (col("gps_lat") != 0)
            & (col("gps_lon").isNotNull())
            & (col("gps_lon") != 0)
        )

        invalid_metadata = bronze_frame_metadata.filter(
            (col("gps_lat").isNull())
            | (col("gps_lat") == 0)
            | (col("gps_lon").isNull())
            | (col("gps_lon") == 0)
        )

        print("02: Processed frame metadata.")

        return valid_metadata, invalid_metadata

    def process_detection_metadata(self, bronze_detection_metadata):

        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata = self.spark.table(
            f"{self.catalog}.{self.schema}.silver_frame_metadata"
        ).alias("silver_frame")
        valid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata,
                F.col("bronze_detection.image_name")
                == F.col("silver_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )

        silver_frame_metadata_quarantine = self.spark.table(
            f"{self.catalog}.{self.schema}.silver_frame_metadata_quarantine"
        ).alias("quarantine_frame")
        invalid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_quarantine,
                F.col("bronze_detection.image_name")
                == F.col("quarantine_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )

        print("02: Processed detection metadata.")

        return valid_metadata, invalid_metadata
