import os
import tempfile

from pyspark.sql.functions import col, lit


class DataLoader:

    def __init__(
        self,
        spark,
        catalog,
        schema,
        root_source,
        device_id,
        ckpt_frames_relative_path,
        ckpt_detections_relative_path,
        job_process_time,
    ):
        self.spark = spark
        self.catalog = catalog
        self.schema = schema
        self.root_source = root_source
        self.device_id = device_id
        self.checkpoint_frames = (
            f"{self.root_source}/{self.device_id}/{ckpt_frames_relative_path}"
        )
        self.checkpoint_detections = (
            f"{self.root_source}/{self.device_id}/{ckpt_detections_relative_path}"
        )
        self.frame_metadata_table = (
            f"{self.catalog}.{self.schema}.bronze_frame_metadata"
        )
        self.detection_metadata_table = (
            f"{self.catalog}.{self.schema}.bronze_detection_metadata"
        )
        self.temp_files = []
        self.job_process_time = job_process_time

    def _get_schema_path(self, table_name):
        """
        Retrieves the schema of the specified table and saves it to a temporary file.

        Parameters:
            table_name (str): The name of the table.

        Returns:
            str: The path to the temporary file containing the schema JSON.
        """
        # Retrieve the schema of the specified table
        existing_table_schema = self.spark.table(table_name).schema
        schema_json = existing_table_schema.json()

        # Save the JSON schema to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write(schema_json)
            path_table_schema = temp_file.name

        self.temp_files.append(path_table_schema)
        return path_table_schema

    def ingest_frame_metadata(self):

        source = f"{self.root_source}/{self.device_id}/frame_metadata"
        path_table_schema = self._get_schema_path(self.frame_metadata_table)
        df = self._load_new_frame_metadata(
            source, path_table_schema=path_table_schema, format="csv"
        )
        print("01: Loaded frame metadata.")
        self._store_new_data(
            df, checkpoint_path=self.checkpoint_frames, target=self.frame_metadata_table
        )

    def ingest_detection_metadata(self):

        source = f"{self.root_source}/{self.device_id}/detection_metadata"
        path_table_schema = self._get_schema_path(self.detection_metadata_table)
        df = self._load_new_detection_metadata(
            source, path_table_schema=path_table_schema, format="csv"
        )
        print("01: Loaded detection metadata.")
        self._store_new_data(
            df,
            checkpoint_path=self.checkpoint_detections,
            target=self.detection_metadata_table,
        )

    def _load_new_frame_metadata(
        self, source: str, path_table_schema: str, format: str
    ):

        bronze_df_frame = (
            self.spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", format)
            .option("cloudFiles.schemaLocation", path_table_schema)
            .option("cloudFiles.inferColumnTypes", "true")
            .option(
                "cloudFiles.schemaHints",
                "timestamp double, imu_pitch float, imu_roll float, imu_heading float, imu_gx float, imu_gy float, imu_gz float, code_version string, gps_lat string, gps_lon string, gps_date string, gps_time timestamp",
            )
            .option("cloudFiles.schemaEvolutionMode", "none")
            .load(source)
            .withColumnRenamed("pylon://0_frame_counter", "pylon0_frame_counter")
            .withColumnRenamed("pylon://0_frame_timestamp", "pylon0_frame_timestamp")
            .withColumn(
                "pylon0_frame_timestamp", col("pylon0_frame_timestamp").cast("double")
            )
            .withColumn("gps_timestamp", col("gps_timestamp").cast("double"))
            .withColumn(
                "gps_internal_timestamp", col("gps_internal_timestamp").cast("double")
            )
            .withColumn("status", lit("Pending"))
        )

        return bronze_df_frame

    def _load_new_detection_metadata(
        self, source: str, path_table_schema: str, format: str
    ):
        bronze_df_detection = (
            self.spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", format)
            .option("cloudFiles.schemaLocation", path_table_schema)
            .option("cloudFiles.inferColumnTypes", "true")
            .option(
                "cloudFiles.schemaHints",
                "x_center float, y_center float, width float, height float, confidence float, tracking_id int",
            )
            .option("cloudFiles.schemaEvolutionMode", "none")
            .load(source)
            .withColumn("status", lit("Pending"))
        )

        return bronze_df_detection

    # availableNow = process all files that have been added before the time when this query ran. Used with batch processing
    def _store_new_data(self, df, checkpoint_path, target):
        stream_query = (
            df.writeStream.option("checkpointLocation", checkpoint_path)
            .trigger(availableNow=True)
            .toTable(target)
        )

        query_progress = stream_query.awaitTermination(60)

        # Get number of rows processed
        if query_progress:
            rows_processed = stream_query.lastProgress["numInputRows"]
            print(f"01: Stored {rows_processed} new rows into {target}.")
        else:
            print("01: Query did not terminate properly.")

    def cleanup_temp_files(self):
        """
        Deletes all temporary files created during the processing.
        """
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"01: Deleted temporary file: {temp_file}")
