from pyspark.sql.functions import col


class MetadataHealthChecker:
    def __init__(self, spark, catalog, schema, job_process_time):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.job_process_time = job_process_time

    def load_bronze_metadata(self, table_name):
        df = self.spark.table(f"{self.catalog}.{self.schema}.{table_name}").filter("status = 'Pending'")
        print(
            f"02: Loaded {df.count()} 'Pending' rows from {self.catalog}.{self.schema}.{table_name}."
        )
        return df

    def process_and_save_frame_metadata(self, bronze_frame_metadata):

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

        valid_metadata.write.mode("append").saveAsTable(
            f"{self.catalog}.{self.schema}.silver_frame_metadata"
        )
        print(
            f"02: Appended {valid_metadata.count()} rows to {self.catalog}.{self.schema}.silver_frame_metadata."
        )

        invalid_metadata.write.mode("append").saveAsTable(
            f"{self.catalog}.{self.schema}.silver_frame_metadata_quarantine"
        )
        print(
            f"02: Appended {invalid_metadata.count()} rows to {self.catalog}.{self.schema}.silver_frame_metadata_quarantine."
        )

    def process_and_save_detection_metadata(self):

        # Detection metadata corresponding to healthy frame metadata is healthy
        valid_metadata_query = f"""
                        SELECT {self.catalog}.{self.schema}.bronze_detection_metadata.*
                        FROM {self.catalog}.{self.schema}.bronze_detection_metadata
                        INNER JOIN {self.catalog}.{self.schema}.silver_frame_metadata ON {self.catalog}.{self.schema}.bronze_detection_metadata.image_name = {self.catalog}.{self.schema}.silver_frame_metadata.image_name
                        WHERE {self.catalog}.{self.schema}.bronze_detection_metadata.status = 'Pending'
                        """
        valid_metadata = self.spark.sql(valid_metadata_query)

        # Detection metadata corresponding to unhealthy frame metadata is unhealthy
        invalid_metadata_query = f"""
                        SELECT {self.catalog}.{self.schema}.bronze_detection_metadata.*
                        FROM {self.catalog}.{self.schema}.bronze_detection_metadata
                        INNER JOIN {self.catalog}.{self.schema}.silver_frame_metadata_quarantine ON {self.catalog}.{self.schema}.bronze_detection_metadata.image_name = {self.catalog}.{self.schema}.silver_frame_metadata_quarantine.image_name
                        WHERE {self.catalog}.{self.schema}.bronze_detection_metadata.status = 'Pending'
                        """

        invalid_metadata = self.spark.sql(invalid_metadata_query)
        print("02: Processed detection metadata.")

        valid_metadata.write.mode("append").saveAsTable(
            f"{self.catalog}.{self.schema}.silver_detection_metadata"
        )
        print(
            f"02: Appended {valid_metadata.count()} rows to silver_detection_metadata."
        )

        invalid_metadata.write.mode("append").saveAsTable(
            f"{self.catalog}.{self.schema}.silver_detection_metadata_quarantine"
        )
        print(
            f"02: Appended {invalid_metadata.count()} rows to silver_detection_metadata_quarantine."
        )
